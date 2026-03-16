import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from tf2_ros import TransformListener, Buffer
import numpy as np
import math
import cv2
from custom_interfaces.srv import TriggerReplan
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from transforms3d import quaternions
from rclpy.executors import MultiThreadedExecutor
from threading import Lock, Thread


class WaypointReplanner(Node):
    def __init__(self):
        super().__init__('waypoint_replanner')
        
        # Initialize locks for thread safety
        self.point_cloud_lock = Lock()
        self.costmap_lock = Lock()
        
        self._declare_parameters()
        
        # Setup TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Initialize point cloud storage
        self.latest_lidar_points = None
        self.latest_camera_points = None
        
        self._setup_communication()
        
        # State variables
        self.local_costmap = None
        self.replan_pending = False
        self.is_replanning = False
        
        self.get_logger().info('Replanning service started')

    def _declare_parameters(self):
        """Declare and load all node parameters"""
        self.declare_parameter('local_range', 20.0)
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('inflation_radius', 0.2)
        self.declare_parameter('height_filter_min', -1.0)
        self.declare_parameter('height_filter_max', 0.5)
        self.declare_parameter('search_radius', 5.0)
        self.declare_parameter('num_radius_samples', 5)
        self.declare_parameter('num_angle_samples', 11)
        self.declare_parameter('max_sample_radius', 5.0)
        self.declare_parameter('angle_range', 0.785398)  # pi/4
        self.declare_parameter('cost_weights', [0.3, 0.6, 0.1])  # [obstacle, smoothness, position]
        self.declare_parameter('visualization_circle_radius', 3)
        self.declare_parameter('visualization_line_thickness', 1)
        self.declare_parameter('legend_height', 60)
        self.declare_parameter('legend_font_scale', 0.5)
        self.declare_parameter('legend_text_thickness', 1)
        self.declare_parameter('enable_visualization', True)
        
        # Load all parameters
        self.local_range = self.get_parameter('local_range').value
        self.resolution = self.get_parameter('resolution').value
        self.inflation_radius = self.get_parameter('inflation_radius').value
        self.height_filter_min = self.get_parameter('height_filter_min').value
        self.height_filter_max = self.get_parameter('height_filter_max').value
        self.search_radius = self.get_parameter('search_radius').value
        self.num_radius_samples = self.get_parameter('num_radius_samples').value
        self.num_angle_samples = self.get_parameter('num_angle_samples').value
        self.max_sample_radius = self.get_parameter('max_sample_radius').value
        self.angle_range = self.get_parameter('angle_range').value
        self.cost_weights = self.get_parameter('cost_weights').value
        self.vis_circle_radius = self.get_parameter('visualization_circle_radius').value
        self.vis_line_thickness = self.get_parameter('visualization_line_thickness').value
        self.legend_height = self.get_parameter('legend_height').value
        self.legend_font_scale = self.get_parameter('legend_font_scale').value
        self.legend_text_thickness = self.get_parameter('legend_text_thickness').value
        self.enable_visualization = self.get_parameter('enable_visualization').value

    def _setup_communication(self):
        """Setup all ROS communications"""
        point_cloud_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            liveliness=rclpy.qos.LivelinessPolicy.AUTOMATIC,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self.lidar_subscriber = self.create_subscription(
            PointCloud2,
            '/segmentation/obstacle',
            self.lidar_callback,
            point_cloud_qos)
            
        self.camera_subscriber = self.create_subscription(
            PointCloud2,
            '/ipm_projected_point_cloud',
            self.camera_callback,
            point_cloud_qos)
            
        # Services
        self.replan_service = self.create_service(
            TriggerReplan,
            'trigger_replan',
            self.handle_replan_request
        )

    def lidar_callback(self, msg):
        """Store latest lidar point cloud"""
        with self.point_cloud_lock:
            self.latest_lidar_points = msg
        self.get_logger().debug('Received lidar points')

    def camera_callback(self, msg):
        """Store latest camera point cloud"""
        with self.point_cloud_lock:
            self.latest_camera_points = msg
        self.get_logger().debug('Received camera points')

    def handle_replan_request(self, request, response):
        """Handle replanning service request
        
        Args:
            request: TriggerReplan request containing waypoints (PoseArray)
            response: TriggerReplan response with success flag and new_waypoints (PoseArray)
        """
        try:
            # Validate input waypoints
            if not request.waypoints.poses:
                self.get_logger().warn('Invalid waypoints received: empty poses')
                response.success = False
                response.new_waypoints = request.waypoints  # Return original path
                return response

            self.target_waypoints = request.waypoints
            success = self.execute_replan()
            
            # Initialize return PoseArray
            response.new_waypoints = PoseArray()
            response.new_waypoints.header = request.waypoints.header
            
            if success and hasattr(self, 'optimized_path'):
                response.success = True
                response.new_waypoints.poses = self.optimized_path
            else:
                response.success = False
                # Return original path if planning fails
                response.new_waypoints.poses = request.waypoints.poses
            
            return response
            
        except Exception as e:
            self.get_logger().error(f'Replan request failed: {str(e)}')
            response.success = False
            # Return original path on error
            response.new_waypoints = request.waypoints
            return response

    def execute_replan(self):
        """Execute replanning process
        
        Returns:
            bool: True if replanning was successful
        """
        self.get_logger().info('Starting replanning process')
        self.local_costmap = self.process_point_clouds()
        if self.local_costmap is None:
            self.get_logger().warn('Local costmap generation failed')
            return False

        try:
            current_transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().error(f'Failed to get transform: {str(e)}')
            return False
        
        current_pose = PoseStamped()
        current_pose.pose.position.x = current_transform.transform.translation.x
        current_pose.pose.position.y = current_transform.transform.translation.y
        current_pose.pose.position.z = current_transform.transform.translation.z
        current_pose.pose.orientation = current_transform.transform.rotation
        
        self.generate_waypoints(current_pose.pose)
        self.get_logger().info('Replanning completed')
        return True

    def process_point_clouds(self):
        """Process point clouds to generate local costmap
        
        Returns:
            np.ndarray: Generated costmap or None if processing fails
        """
        with self.point_cloud_lock:
            lidar_msg = self.latest_lidar_points
            camera_msg = self.latest_camera_points

        if not lidar_msg and not camera_msg:
            self.get_logger().debug('Waiting for point cloud data...')
            return None

        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)
            )
                
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y

            map_size = int(2 * self.local_range / self.resolution)
            local_costmap = np.zeros((map_size, map_size), dtype=np.int8)
            
            def process_points(points_msg, transform):
                if points_msg is None:
                    return np.array([])
                    
                try:
                    pc_array = np.array(list(pc2.read_points(points_msg, 
                                                           field_names=("x", "y", "z"), 
                                                           skip_nans=True)))
                    
                    if pc_array.size == 0:
                        return np.array([])

                    points = np.stack((pc_array['x'], pc_array['y'], pc_array['z']), axis=-1).astype(np.float32)
                    
                    quat = [
                        transform.transform.rotation.w,
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z
                    ]
                    
                    rot_matrix = quaternions.quat2mat(quat)
                    translation = np.array([
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z
                    ])
                    
                    transformed_points = points @ rot_matrix.T + translation
                    
                    height_mask = (transformed_points[:, 2] >= self.height_filter_min) & \
                                 (transformed_points[:, 2] <= self.height_filter_max)
                    filtered_points = transformed_points[height_mask]
                    
                    return filtered_points[:, :2]
                    
                except Exception as e:
                    self.get_logger().error(f'Point cloud processing failed: {str(e)}')
                    return np.array([])
            
            # Process lidar point cloud
            lidar_points_map = np.array([])
            if lidar_msg:
                try:
                    lidar_transform = self.tf_buffer.lookup_transform(
                        'map',
                        lidar_msg.header.frame_id,
                        lidar_msg.header.stamp,
                        rclpy.duration.Duration(seconds=0.1)
                    )
                    lidar_points_map = process_points(lidar_msg, lidar_transform)
                except Exception as e:
                    self.get_logger().error(f'Lidar processing failed: {str(e)}')
            
            # Process camera point cloud
            camera_points_map = np.array([])
            if camera_msg:
                try:
                    camera_transform = self.tf_buffer.lookup_transform(
                        'map',
                        camera_msg.header.frame_id,
                        camera_msg.header.stamp,
                        rclpy.duration.Duration(seconds=0.1)
                    )
                    camera_points_map = process_points(camera_msg, camera_transform)
                except Exception as e:
                    self.get_logger().error(f'Camera processing failed: {str(e)}')

            # Add points to costmap
            def add_points_to_costmap(points, costmap):
                if points.size == 0:
                    return
                    
                rel_points = points - np.array([robot_x, robot_y])
                mask = np.all(np.abs(rel_points) < self.local_range, axis=1)
                valid_points = rel_points[mask]
                
                if valid_points.size > 0:
                    map_coords = ((valid_points + self.local_range) / self.resolution).astype(int)
                    valid_indices = np.logical_and(
                        map_coords[:, 0] >= 0, map_coords[:, 0] < map_size
                    ) & np.logical_and(
                        map_coords[:, 1] >= 0, map_coords[:, 1] < map_size
                    )
                    map_coords = map_coords[valid_indices]
                    costmap[map_coords[:, 1], map_coords[:, 0]] = 100

            add_points_to_costmap(lidar_points_map, local_costmap)
            add_points_to_costmap(camera_points_map, local_costmap)
            
            # Perform obstacle inflation
            kernel_size = max(1, int(self.inflation_radius / self.resolution))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1))
            obstacle_layer = (local_costmap == 100).astype(np.uint8)
            
            dist_transform = cv2.distanceTransform(1 - obstacle_layer, cv2.DIST_L2, 5)
            max_dist = kernel_size * self.resolution
            cost_from_distance = np.where(
                dist_transform < max_dist,
                (100 * (1 - dist_transform / max_dist)).astype(np.int8),
                0
            )
            
            local_costmap = np.maximum(local_costmap, cost_from_distance)
            
            return local_costmap

        except Exception as e:
            self.get_logger().error(f'Point cloud processing failed: {str(e)}')
            return None

    def generate_waypoints(self, robot_pose):
        """Generate optimized path
        
        Args:
            robot_pose: Current robot pose
        """
        if not hasattr(self, 'target_waypoints') or not self.target_waypoints.poses:
            self.get_logger().warn('No waypoints available for replanning')
            return
        
        self.current_position = robot_pose
        optimized_poses = []
        optimized_poses.append(self.target_waypoints.poses[0])

        if len(self.target_waypoints.poses) >= 2:
            start_pose = self.target_waypoints.poses[0]
            end_pose = self.target_waypoints.poses[1]
            
            best_wp = self.optimize_waypoint(start_pose, end_pose)
            
            if best_wp:
                # Calculate offset from the optimized second waypoint
                dx = best_wp[0] - end_pose.position.x
                dy = best_wp[1] - end_pose.position.y
                
                # Add the optimized second waypoint
                new_pose = Pose()
                new_pose.position.x = best_wp[0]
                new_pose.position.y = best_wp[1]
                new_pose.position.z = 0.0
                new_pose.orientation = end_pose.orientation
                optimized_poses.append(new_pose)
                
                # Apply the same offset to remaining waypoints
                for i in range(2, len(self.target_waypoints.poses)):
                    original_pose = self.target_waypoints.poses[i]
                    new_pose = Pose()
                    new_pose.position.x = original_pose.position.x + dx
                    new_pose.position.y = original_pose.position.y + dy
                    new_pose.position.z = original_pose.position.z
                    new_pose.orientation = original_pose.orientation
                    optimized_poses.append(new_pose)
        
        # Store optimized path for service response
        self.optimized_path = optimized_poses
        
        # Only perform visualization if enabled
        if self.enable_visualization:
            try:
                transform = self.tf_buffer.lookup_transform(
                    'map',
                    'base_link',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                    
                robot_x = int((transform.transform.translation.x + self.local_range) / self.resolution)
                robot_y = int((transform.transform.translation.y + self.local_range) / self.resolution)
                
                self.visualize_costmap(
                    self.local_costmap, 
                    robot_x, 
                    robot_y,
                    original_path=self.target_waypoints.poses,
                    optimized_path=optimized_poses
                )
            except Exception as e:
                self.get_logger().error(f'Failed to visualize costmap: {str(e)}')

    def distance_to_obstacle(self, x, y):
        """Calculate actual distance to nearest obstacle
        
        Args:
            x: Global x coordinate
            y: Global y coordinate
            
        Returns:
            float: Distance to nearest obstacle in meters
        """
        if self.local_costmap is None:
            return float('inf')
        
        center_x = int((x + self.local_range) / self.resolution)
        center_y = int((y + self.local_range) / self.resolution)
        
        if not (0 <= center_x < self.local_costmap.shape[1] and 0 <= center_y < self.local_costmap.shape[0]):
            return 0.0
        
        search_radius = int(self.search_radius / self.resolution)
        min_x = max(0, center_x - search_radius)
        max_x = min(self.local_costmap.shape[1], center_x + search_radius + 1)
        min_y = max(0, center_y - search_radius)
        max_y = min(self.local_costmap.shape[0], center_y + search_radius + 1)
        
        obstacle_points = []
        local_map_slice = self.local_costmap[min_y:max_y, min_x:max_x]
        obstacle_indices = np.argwhere(local_map_slice == 100)
        
        for idx in obstacle_indices:
            dx = (idx[1] + min_x - center_x) * self.resolution
            dy = (idx[0] + min_y - center_y) * self.resolution
            dist = math.sqrt(dx*dx + dy*dy)
            obstacle_points.append(dist)
        
        return min(obstacle_points) if obstacle_points else 5.0

    def distance(self, x1, y1, x2, y2):
        """Calculate Euclidean distance between two points"""
        return math.hypot(x2 - x1, y2 - y1)

    def optimize_waypoint(self, start, end):
        """Optimize waypoint position within a sector-shaped sampling area
        
        Args:
            start: Starting pose
            end: Target pose
            
        Returns:
            tuple: Optimized waypoint coordinates (x, y)
        """
        dx = end.position.x - start.position.x
        dy = end.position.y - start.position.y
        target_angle = math.atan2(dy, dx)
        target_distance = self.distance(start.position.x, start.position.y, end.position.x, end.position.y)
        
        num_radius_samples = self.num_radius_samples
        num_angle_samples = self.num_angle_samples
        max_radius = min(self.max_sample_radius, target_distance)
        angle_range = self.angle_range
        
        best_point = None
        min_score = float('inf')
        
        radius_step = max_radius / num_radius_samples
        angle_step = (2 * angle_range) / (num_angle_samples - 1)
        
        for i in range(1, num_radius_samples + 1):
            radius = i * radius_step
            
            for j in range(num_angle_samples):
                angle = target_angle - angle_range + j * angle_step
                sample_x = self.current_position.position.x + radius * math.cos(angle)
                sample_y = self.current_position.position.y + radius * math.sin(angle)
                
                obstacle_cost = 1.0 - self.distance_to_obstacle(sample_x, sample_y) / max_radius
                path_angle = math.atan2(
                    sample_y - self.current_position.position.y,
                    sample_x - self.current_position.position.x
                )
                orig_angle = math.atan2(
                    dy,
                    dx
                )
                smoothness_cost = abs(path_angle - orig_angle) / self.angle_range
                position_cost = 1.0 - self.distance(
                    self.current_position.position.x,
                    self.current_position.position.y,
                    sample_x, sample_y
                ) / max_radius
                
                total_cost = (
                    self.cost_weights[0] * obstacle_cost +
                    self.cost_weights[1] * smoothness_cost +
                    self.cost_weights[2] * position_cost
                )
                
                if total_cost < min_score:
                    min_score = total_cost
                    best_point = (sample_x, sample_y)
                    
                self.get_logger().debug(
                    f'Sample point ({sample_x:.2f}, {sample_y:.2f}): '
                    f'radius={radius:.2f}, angle={math.degrees(angle):.1f}°, '
                    f'obstacle_cost={obstacle_cost:.2f}, '
                    f'smoothness_cost={smoothness_cost:.2f}, '
                    f'position_cost={position_cost:.2f}, '
                    f'total_cost={total_cost:.2f}'
                )
        
        if best_point is None:
            self.get_logger().warn('No valid sample point found, using midpoint')
            best_point = (
                start.position.x + 0.5 * target_distance * math.cos(target_angle),
                start.position.y + 0.5 * target_distance * math.sin(target_angle)
            )
        
        self.get_logger().info(
            f'Original point: ({end.position.x:.2f}, {end.position.y:.2f})\n'
            f'Best point found at ({best_point[0]:.2f}, {best_point[1]:.2f}) '
            f'with score {min_score:.2f}'
        )
        
        return best_point

    def visualize_costmap(self, local_costmap, robot_x, robot_y, original_path=None, optimized_path=None):
        """Visualize costmap and paths
        
        Args:
            local_costmap: 2D costmap array
            robot_x: Robot x position in costmap coordinates
            robot_y: Robot y position in costmap coordinates
            original_path: Original path points (optional)
            optimized_path: Optimized path points (optional)
        """
        try:
            if local_costmap is None or local_costmap.size == 0:
                self.get_logger().warn('Local costmap is empty, cannot visualize')
                return
            
            if not (0 <= robot_x < local_costmap.shape[1] and 0 <= robot_y < local_costmap.shape[0]):
                self.get_logger().warn('Robot position outside local map bounds')
                return
            
            vis_map = np.zeros_like(local_costmap, dtype=np.uint8)
            vis_map[local_costmap == -1] = 128
            mask = (local_costmap >= 0) & (local_costmap <= 100)
            vis_map[mask] = (local_costmap[mask] * 255 / 100).astype(np.uint8)
            
            vis_map_color = cv2.cvtColor(vis_map, cv2.COLOR_GRAY2BGR)
            cv2.circle(vis_map_color, (int(robot_x), int(robot_y)), 
                       self.vis_circle_radius, (0, 0, 255), -1)
            
            def to_map_coords(x, y):
                map_x = int((x + self.local_range) / self.resolution)
                map_y = int((y + self.local_range) / self.resolution)
                return map_x, map_y
            
            self._draw_path(original_path, vis_map_color, to_map_coords, (255, 0, 0))
            self._draw_path(optimized_path, vis_map_color, to_map_coords, (0, 255, 0))
            
            legend_height = self.legend_height
            legend = np.ones((legend_height, vis_map_color.shape[1], 3), dtype=np.uint8) * 255
            
            cv2.putText(legend, "Original Path", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        self.legend_font_scale, (255, 0, 0), self.legend_text_thickness)
            cv2.putText(legend, "Optimized Path", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        self.legend_font_scale, (0, 255, 0), self.legend_text_thickness)
            
            vis_map_with_legend = np.vstack([vis_map_color, legend])
            
            cv2.imshow('Local Costmap with Paths', vis_map_with_legend)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Failed to visualize costmap: {str(e)}')

    def _draw_path(self, path, vis_map, to_map_coords, color):
        """Helper method to draw path on visualization map
        
        Args:
            path: List of poses (PoseStamped or Pose)
            vis_map: Visualization map to draw on
            to_map_coords: Function to convert world coordinates to map coordinates
            color: Color to draw the path (B,G,R)
        """
        if not path:
            return
            
        for i, pose in enumerate(path):
            # Handle both PoseStamped and Pose objects
            if hasattr(pose, 'pose'):  # PoseStamped
                position = pose.pose.position
            else:  # Pose
                position = pose.position
            
            map_x, map_y = to_map_coords(position.x, position.y)
            
            if 0 <= map_x < vis_map.shape[1] and 0 <= map_y < vis_map.shape[0]:
                cv2.circle(vis_map, (map_x, map_y), 
                          self.vis_circle_radius, color, -1)
                
                if i > 0:
                    prev_pose = path[i-1]
                    if hasattr(prev_pose, 'pose'):  # PoseStamped
                        prev_position = prev_pose.pose.position
                    else:  # Pose
                        prev_position = prev_pose.position
                    
                    prev_map_x, prev_map_y = to_map_coords(prev_position.x, prev_position.y)
                    if (0 <= prev_map_x < vis_map.shape[1] and 
                        0 <= prev_map_y < vis_map.shape[0]):
                        cv2.line(vis_map, (prev_map_x, prev_map_y), (map_x, map_y), 
                                 color, self.vis_line_thickness)


def main(args=None):
    rclpy.init(args=args)
    replanner = WaypointReplanner()
    executor = MultiThreadedExecutor()
    
    try:
        executor.add_node(replanner)
        executor.spin()
    finally:
        replanner.destroy_node()
        rclpy.shutdown()
        executor.shutdown()


if __name__ == '__main__':
    main()
