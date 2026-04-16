import time
from math import radians, cos, sin, atan2, sqrt

# Third-party imports
import numpy as np
import osmium as osm
import pyclipper

# ROS imports
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, PoseStamped, Quaternion
from sensor_msgs.msg import Image
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

# Custom imports
from custom_interfaces.action import EntranceExploration
from custom_interfaces.srv import GetEntranceId
from utils_pkg import OSMHandler, CoordinateTransformer

class EntranceExplorationActionServer(Node):
    def __init__(self):
        super().__init__('entrance_exploration_action_server')
        
        # Load parameters
        self._load_parameters()
        
        # Initialize components
        self.osm_handler = OSMHandler()
        self.osm_handler.apply_file(self.osm_file_path)
        self.callback_group = ReentrantCallbackGroup()
        
        # Initialize core components
        self._init_action_server()
        self._init_navigation()
        self._init_perception()
        self._init_state_variables()

    def _load_parameters(self):
        self.declare_parameters(
            namespace='',
            parameters=[
                ('osm_file_path', '/workspaces/light-map-navigation/src/llm_exploration_py/OSM/medium.osm'),
                ('exploration_radius', 2.0),
                ('exploration_points', 5),
                ('exploration_strategy', 'baseline'),
                ('mfvs_distance_weight', 0.45),
                ('mfvs_saliency_weight', 0.32),
                ('mfvs_coverage_weight', 0.10),
                ('mfvs_novelty_weight', 0.05),
                ('mfvs_heading_weight', 0.04),
                ('mfvs_corner_weight', 0.04),
                ('mfvs_corner_saliency_threshold', 0.45),
                ('mfvs_candidate_offset_fraction', 0.35),
                ('mfvs_pruning_distance_factor', 0.75),
                ('camera_topic', '/camera_sensor/image_raw'),
                ('map_frame', 'map'),
                ('bk_frame', 'base_link'),
                ('transform_matrix', [
                    1.0, 0.0, -500000.0,
                    0.0, 1.0, -4483000.0,
                    0.0, 0.0, 1.0
                ]),
                ('navigation_feedback_interval', 2.0),
                ('service_timeout', 1.0)
            ]
        )
        
        # Store parameters as instance variables
        self.osm_file_path = self.get_parameter('osm_file_path').value
        self.exploration_radius = self.get_parameter('exploration_radius').value
        self.exploration_points = self.get_parameter('exploration_points').value
        self.exploration_strategy = str(self.get_parameter('exploration_strategy').value).strip().lower()
        self.mfvs_distance_weight = float(self.get_parameter('mfvs_distance_weight').value)
        self.mfvs_saliency_weight = float(self.get_parameter('mfvs_saliency_weight').value)
        self.mfvs_coverage_weight = float(self.get_parameter('mfvs_coverage_weight').value)
        self.mfvs_novelty_weight = float(self.get_parameter('mfvs_novelty_weight').value)
        self.mfvs_heading_weight = float(self.get_parameter('mfvs_heading_weight').value)
        self.mfvs_corner_weight = float(self.get_parameter('mfvs_corner_weight').value)
        self.mfvs_corner_saliency_threshold = float(self.get_parameter('mfvs_corner_saliency_threshold').value)
        self.mfvs_candidate_offset_fraction = float(self.get_parameter('mfvs_candidate_offset_fraction').value)
        self.mfvs_pruning_distance_factor = float(self.get_parameter('mfvs_pruning_distance_factor').value)
        self.adaptive_distance_weight = self.mfvs_distance_weight
        self.adaptive_diversity_weight = self.mfvs_saliency_weight
        self.mfvs_diversity_weight = self.mfvs_saliency_weight
        self.camera_topic = self.get_parameter('camera_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.bk_frame = self.get_parameter('bk_frame').value

        if self.exploration_strategy == 'adaptive':
            self.exploration_strategy = 'mfvs'

        if self.exploration_strategy not in ('baseline', 'mfvs'):
            self.get_logger().warn(
                f"Unsupported exploration_strategy '{self.exploration_strategy}', falling back to baseline"
            )
            self.exploration_strategy = 'baseline'
        
        # Get new parameters
        transform_matrix_list = self.get_parameter('transform_matrix').value
        self.transform_matrix = np.array(transform_matrix_list).reshape(3, 3)
        self.navigation_feedback_interval = self.get_parameter('navigation_feedback_interval').value
        self.service_timeout = self.get_parameter('service_timeout').value

    def _init_action_server(self):
        self._action_server = ActionServer(
            self,
            EntranceExploration,
            'explore_entrance',
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )

    def _init_navigation(self):
        """Initialize navigation-related components"""
        self.navigator = BasicNavigator()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
    def _init_perception(self):
        """Initialize perception-related components"""
        # Service client
        self.get_entrance_id_client = self.create_client(GetEntranceId, 'entrance_recognition')
        self._wait_for_service()
        
        # Image subscriber
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10)
            
    def _init_state_variables(self):
        """Initialize state tracking variables"""
        self.latest_image = None
        self.cur_position = None
        self.current_robot_heading = 0.0
        self.current_building_center = None
        self.current_waypoint_index = 0
        self.waypoint_ = []
        self.mfvs_candidates = []
        self.remaining_waypoint_indices = []
        self.mfvs_remaining_indices = []
        self.mfvs_visited_positions = []
        self.mfvs_visited_arc_positions = []
        self.mfvs_visited_keys = set()
        self.mfvs_last_goal_pose = None
        self.mfvs_perimeter_length = 0.0
        self.visited_waypoint_indices = set()
        self.current_exploration_step = 0
        self.is_task_success = False
        self._cancel_requested = False
        self._current_goal_handle = None

    def _reset_state(self):
        """Reset all state variables for new goal"""
        self.current_waypoint_index = 0
        self.waypoint_ = []
        self.current_robot_heading = 0.0
        self.current_building_center = None
        self.mfvs_candidates = []
        self.remaining_waypoint_indices = []
        self.mfvs_remaining_indices = []
        self.mfvs_visited_positions = []
        self.mfvs_visited_arc_positions = []
        self.mfvs_visited_keys = set()
        self.mfvs_last_goal_pose = None
        self.mfvs_perimeter_length = 0.0
        self.visited_waypoint_indices = set()
        self.current_exploration_step = 0
        self.is_task_success = False
        self._cancel_requested = False
        self._current_goal_handle = None

    def _wait_for_service(self):
        """Wait for required services to become available"""
        while not self.get_entrance_id_client.wait_for_service(timeout_sec=self.service_timeout):
            self.get_logger().info('Waiting for entrance_recognition service...')

    def image_callback(self, msg):
        """Callback function for receiving image data"""
        self.latest_image = msg
            
    def goal_callback(self, goal_request):
        """
        Callback for handling new goal requests
        Args:
            goal_request: The goal request message
        Returns:
            GoalResponse: Accept or reject the goal request
        """
        self.get_logger().info('Received goal request')
        
        # Validate goal parameters
        try:
            if not isinstance(goal_request.building_id, str) or not isinstance(goal_request.unit_id, str):
                self.get_logger().error('Invalid goal: building_id and unit_id must be strings')
                return GoalResponse.REJECT
                
            if not goal_request.building_id or not goal_request.unit_id:
                self.get_logger().error('Invalid goal: building_id and unit_id cannot be empty')
                return GoalResponse.REJECT
                
            # If there is an active task, reject the new request
            if hasattr(self, '_current_goal_handle') and self._current_goal_handle is not None:
                self.get_logger().warn('Another goal is already active, rejecting new goal')
                return GoalResponse.REJECT
                
            self.get_logger().info(f'Goal accepted - Building: {goal_request.building_id}, Unit: {goal_request.unit_id}')
            return GoalResponse.ACCEPT
            
        except Exception as e:
            self.get_logger().error(f'Error in goal validation: {str(e)}')
            return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        """
        Callback for handling cancellation requests
        Args:
            goal_handle: The goal handle being cancelled
        Returns:
            CancelResponse: Accept or reject the cancel request
        """
        self.get_logger().info('Received cancel request')
        
        try:
            if not hasattr(self, '_current_goal_handle') or self._current_goal_handle is None:
                self.get_logger().warn('No active goal to cancel')
                return CancelResponse.REJECT
                
            if goal_handle != self._current_goal_handle:
                self.get_logger().warn('Cancel request does not match current goal')
                return CancelResponse.REJECT
                
            self._cancel_requested = True
            
            # cancel navigation task
            if hasattr(self, 'navigator'):
                self.navigator.cancelTask()
                
            self.get_logger().info('Cancel request accepted')
            return CancelResponse.ACCEPT
            
        except Exception as e:
            self.get_logger().error(f'Error in cancel callback: {str(e)}')
            return CancelResponse.REJECT

    def inflate_building(self, coordinates, offset_distance):
        """Inflate the building polygon by a given offset distance."""
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(coordinates, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        inflated_polygon = pco.Execute(offset_distance)
        return inflated_polygon[0] if inflated_polygon else coordinates

    def sample_waypoints_evenly(self, coordinates, distance):
        """Sample waypoints evenly along the given coordinates."""
        total_length = 0
        num_points = len(coordinates)
        for i in range(num_points):
            start = np.array(coordinates[i])
            end = np.array(coordinates[(i + 1) % num_points])
            total_length += np.linalg.norm(end - start)

        sampled_waypoints = []
        accumulated_distance = 0

        for i in range(num_points):
            start = np.array(coordinates[i])
            end = np.array(coordinates[(i + 1) % num_points])
            segment_length = np.linalg.norm(end - start)

            while accumulated_distance + segment_length >= len(sampled_waypoints) * distance:
                ratio = (len(sampled_waypoints) * distance - accumulated_distance) / segment_length
                new_point = start + ratio * (end - start)
                sampled_waypoints.append(tuple(new_point))

            accumulated_distance += segment_length

        return sampled_waypoints

    def calculate_yaw(self, waypoint, target):
        """Calculate the yaw angle from a waypoint to a target."""
        vector_to_target = np.array(target) - np.array(waypoint)
        yaw = atan2(vector_to_target[1], vector_to_target[0])
        return yaw

    def calculate_orientations(self, all_waypoints, center):
        """Calculate orientations for all waypoints relative to a center point."""
        return [self.calculate_yaw(wp, center) for wp in all_waypoints]

    def get_closest_waypoint_index(self, waypoints, robot_position):
        """Find the index of the closest waypoint to the robot's current position."""
        distances = [sqrt((x - robot_position[0])**2 + (y - robot_position[1])**2) for x, y, yaw in waypoints]
        return distances.index(min(distances))

    def reorder_waypoints(self, waypoints, start_index):
        """Reorder waypoints starting from a specific index."""
        return waypoints[start_index:] + waypoints[:start_index]

    def _quaternion_to_yaw(self, quaternion):
        siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return atan2(siny_cosp, cosy_cosp)

    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _angular_distance(self, angle_a, angle_b):
        return abs(self._normalize_angle(angle_a - angle_b))

    async def _get_current_robot_state(self):
        try:
            transform: TransformStamped = await self.tf_buffer.lookup_transform_async(
                self.map_frame,
                self.bk_frame,
                rclpy.time.Time())
            self.cur_position = (
                transform.transform.translation.x,
                transform.transform.translation.y,
            )
            self.current_robot_heading = self._quaternion_to_yaw(transform.transform.rotation)
        except Exception as e:
            self.get_logger().warn(f'Failed to refresh robot pose, using last known pose: {str(e)}')
        return self.cur_position, self.current_robot_heading

    async def _get_current_robot_position(self):
        """Backward-compatible alias for older call sites."""
        robot_position, _ = await self._get_current_robot_state()
        return robot_position

    def _get_waypoint_heading(self, waypoint):
        if self.current_building_center is None:
            return 0.0
        return atan2(
            waypoint[1] - self.current_building_center[1],
            waypoint[0] - self.current_building_center[0]
        )

    def _build_boundary_geometry(self, coordinates):
        points = [np.asarray(point, dtype=float) for point in coordinates]
        if len(points) > 1 and np.allclose(points[0], points[-1]):
            points = points[:-1]

        if len(points) < 2:
            return points, points, [], [0.0]

        closed_points = points + [points[0]]
        segment_lengths = []
        cumulative_lengths = [0.0]

        for index in range(len(points)):
            segment_length = float(np.linalg.norm(closed_points[index + 1] - closed_points[index]))
            segment_lengths.append(segment_length)
            cumulative_lengths.append(cumulative_lengths[-1] + segment_length)

        return points, closed_points, segment_lengths, cumulative_lengths

    def _calculate_vertex_saliency(self, coordinates):
        if len(coordinates) < 3:
            return [0.0 for _ in coordinates]

        points = [np.asarray(point, dtype=float) for point in coordinates]
        if np.allclose(points[0], points[-1]):
            points = points[:-1]

        saliency = []
        for index in range(len(points)):
            prev_point = points[index - 1]
            current_point = points[index]
            next_point = points[(index + 1) % len(points)]

            vec_prev = prev_point - current_point
            vec_next = next_point - current_point
            prev_norm = np.linalg.norm(vec_prev)
            next_norm = np.linalg.norm(vec_next)

            if prev_norm == 0.0 or next_norm == 0.0:
                saliency.append(0.0)
                continue

            cosine = np.clip(np.dot(vec_prev, vec_next) / (prev_norm * next_norm), -1.0, 1.0)
            interior_angle = np.arccos(cosine)
            saliency.append(float(np.pi - interior_angle))

        max_saliency = max(saliency) if saliency else 0.0
        if max_saliency <= 0.0:
            return [0.0 for _ in saliency]

        return [value / max_saliency for value in saliency]

    def _point_on_boundary(self, closed_points, segment_lengths, cumulative_lengths, target_distance):
        perimeter = cumulative_lengths[-1]
        if perimeter == 0.0:
            return tuple(closed_points[0]), 0.0, 0

        arc_distance = target_distance % perimeter
        segment_index = 0
        while segment_index < len(segment_lengths) - 1 and cumulative_lengths[segment_index + 1] < arc_distance:
            segment_index += 1

        segment_start = closed_points[segment_index]
        segment_end = closed_points[segment_index + 1]
        segment_length = segment_lengths[segment_index]
        if segment_length == 0.0:
            return tuple(segment_start), arc_distance, segment_index

        ratio = (arc_distance - cumulative_lengths[segment_index]) / segment_length
        ratio = np.clip(ratio, 0.0, 1.0)
        sampled_point = segment_start + ratio * (segment_end - segment_start)
        return tuple(sampled_point), arc_distance, segment_index

    def _deduplicate_candidates(self, candidates):
        deduplicated = []
        seen_keys = set()
        dedup_resolution = max(0.25, 0.1 * max(self.exploration_points, self.exploration_radius, 1.0))
        prune_radius = max(self.exploration_points * self.mfvs_pruning_distance_factor, self.exploration_radius * self.mfvs_pruning_distance_factor, 1.0)

        for candidate in sorted(candidates, key=lambda item: (-item['saliency'], item['arc_length'])):
            pose_x, pose_y, _ = candidate['pose']
            candidate_key = (round(pose_x / dedup_resolution), round(pose_y / dedup_resolution))
            if candidate_key in seen_keys:
                continue
            if any(
                sqrt((pose_x - existing['pose'][0]) ** 2 + (pose_y - existing['pose'][1]) ** 2) < prune_radius * 0.45
                for existing in deduplicated
            ):
                continue
            seen_keys.add(candidate_key)
            deduplicated.append(candidate)

        return deduplicated

    def _build_candidate_pool_mfvs(self, inflated_building, sample_distance):
        polygon_points, closed_points, segment_lengths, cumulative_lengths = self._build_boundary_geometry(inflated_building)
        perimeter = cumulative_lengths[-1]
        if perimeter == 0.0:
            return []

        base_step = max(sample_distance, 1e-6)
        fine_step = max(base_step * 0.5, self.exploration_radius)
        candidate_distances = list(np.arange(0.0, perimeter, base_step))
        candidate_distances += list(np.arange(base_step * 0.5, perimeter, fine_step))

        center = np.mean(np.asarray(inflated_building), axis=0)
        vertex_saliency = self._calculate_vertex_saliency(inflated_building)
        candidates = []

        for target_distance in candidate_distances:
            waypoint, arc_length, segment_index = self._point_on_boundary(
                closed_points,
                segment_lengths,
                cumulative_lengths,
                target_distance,
            )
            yaw = self.calculate_yaw(waypoint, center)
            nearest_vertex_index = min(
                range(len(polygon_points)),
                key=lambda index: np.linalg.norm(
                    np.asarray(polygon_points[index], dtype=float) - np.asarray(waypoint, dtype=float)
                ),
            )
            saliency = vertex_saliency[nearest_vertex_index] if vertex_saliency else 0.0
            candidates.append({
                'pose': (float(waypoint[0]), float(waypoint[1]), float(yaw)),
                'saliency': float(saliency),
                'arc_length': float(arc_length),
                'segment_length': float(segment_lengths[segment_index] if segment_index < len(segment_lengths) else 0.0),
                'corner_focus': bool(saliency >= self.mfvs_corner_saliency_threshold),
            })

        if candidates:
            salient_vertices = [index for index, value in enumerate(vertex_saliency) if value >= self.mfvs_corner_saliency_threshold]
            for vertex_index in salient_vertices:
                vertex_arc = cumulative_lengths[min(vertex_index, len(cumulative_lengths) - 2)]
                for offset_ratio in (-self.mfvs_candidate_offset_fraction, 0.0, self.mfvs_candidate_offset_fraction):
                    waypoint, arc_length, segment_index = self._point_on_boundary(
                        closed_points,
                        segment_lengths,
                        cumulative_lengths,
                        vertex_arc + offset_ratio * base_step,
                    )
                    yaw = self.calculate_yaw(waypoint, center)
                    candidates.append({
                        'pose': (float(waypoint[0]), float(waypoint[1]), float(yaw)),
                        'saliency': float(vertex_saliency[vertex_index]),
                        'arc_length': float(arc_length),
                        'segment_length': float(segment_lengths[segment_index] if segment_index < len(segment_lengths) else 0.0),
                        'corner_focus': True,
                    })

        deduplicated_candidates = self._deduplicate_candidates(candidates)
        self.get_logger().info(
            f"MFVS candidate pool built: raw={len(candidates)}, deduplicated={len(deduplicated_candidates)}, "
            f"corner_threshold={self.mfvs_corner_saliency_threshold:.2f}"
        )
        return deduplicated_candidates

    def _score_mfvs_candidate(self, candidate, robot_position, robot_heading):
        candidate_x, candidate_y, candidate_yaw = candidate['pose']
        distance = sqrt((candidate_x - robot_position[0])**2 + (candidate_y - robot_position[1])**2)
        distance_scale = max(self.exploration_points, self.exploration_radius, 1.0)
        if distance < max(0.75, 0.15 * distance_scale):
            return float('-inf')
        distance_score = min(distance / distance_scale, 1.0)

        if self.mfvs_visited_arc_positions:
            arc_deltas = [abs(candidate['arc_length'] - arc) for arc in self.mfvs_visited_arc_positions]
            coverage_gap = min(min(delta, self.mfvs_perimeter_length - delta) for delta in arc_deltas) / (0.5 * max(self.mfvs_perimeter_length, 1e-6))
            coverage_score = max(0.0, min(coverage_gap, 1.0))
        else:
            coverage_score = 1.0

        if self.mfvs_visited_positions:
            nearest_visit = min(
                sqrt((candidate_x - visited_x) ** 2 + (candidate_y - visited_y) ** 2)
                for visited_x, visited_y in self.mfvs_visited_positions
            )
            novelty_score = max(0.0, min(nearest_visit / max(distance_scale, 1e-6), 1.0))
        else:
            novelty_score = 1.0

        heading_error = self._angular_distance(robot_heading, candidate_yaw)
        heading_score = 1.0 - min(heading_error / np.pi, 1.0)
        corner_score = 1.0 if candidate.get('corner_focus', False) else 0.0
        saliency_score = candidate.get('saliency', 0.0)

        return (
            self.mfvs_distance_weight * distance_score
            + self.mfvs_saliency_weight * saliency_score
            + self.mfvs_coverage_weight * coverage_score
            + self.mfvs_novelty_weight * novelty_score
            + self.mfvs_heading_weight * heading_score
            + self.mfvs_corner_weight * corner_score
        )

    def _mfvs_candidate_key(self, candidate):
        pose_x, pose_y, _ = candidate['pose']
        resolution = max(0.25, 0.1 * max(self.exploration_points, self.exploration_radius, 1.0))
        return (round(pose_x / resolution), round(pose_y / resolution))

    def _mfvs_discard_candidate(self, selected_candidate_index):
        selected_candidate = self.mfvs_candidates[selected_candidate_index]
        selected_pose = selected_candidate['pose']
        selected_key = self._mfvs_candidate_key(selected_candidate)
        prune_radius = max(
            1.0,
            self.exploration_points * self.mfvs_pruning_distance_factor,
            self.exploration_radius * self.mfvs_pruning_distance_factor,
        )

        self.get_logger().info(
            f"MFVS discard: index={selected_candidate_index}, pose=({selected_pose[0]:.2f}, {selected_pose[1]:.2f}), "
            f"key={selected_key}, prune_radius={prune_radius:.2f}, remaining_before={len(self.mfvs_remaining_indices)}"
        )
        self.mfvs_visited_keys.add(selected_key)
        self.mfvs_remaining_indices = [
            candidate_index
            for candidate_index in self.mfvs_remaining_indices
            if candidate_index != selected_candidate_index
            and self._mfvs_candidate_key(self.mfvs_candidates[candidate_index]) != selected_key
            and sqrt(
                (self.mfvs_candidates[candidate_index]['pose'][0] - selected_pose[0]) ** 2 +
                (self.mfvs_candidates[candidate_index]['pose'][1] - selected_pose[1]) ** 2
            ) >= prune_radius
        ]
        self.get_logger().info(f"MFVS discard done: remaining_after={len(self.mfvs_remaining_indices)}")

    async def _select_next_waypoint_index(self):
        if self.exploration_strategy == 'baseline':
            if not self.remaining_waypoint_indices:
                return None
            return self.remaining_waypoint_indices[0]

        if not self.mfvs_remaining_indices:
            return None

        robot_position, robot_heading = await self._get_current_robot_state()
        if robot_position is None:
            robot_position = self.cur_position
        if robot_position is None:
            return self.mfvs_remaining_indices[0]

        min_goal_separation = max(1.5, self.exploration_radius * 0.75)
        available_indices = [
            index for index in self.mfvs_remaining_indices
            if self._mfvs_candidate_key(self.mfvs_candidates[index]) not in self.mfvs_visited_keys
            and (
                self.mfvs_last_goal_pose is None
                or sqrt(
                    (self.mfvs_candidates[index]['pose'][0] - self.mfvs_last_goal_pose[0]) ** 2 +
                    (self.mfvs_candidates[index]['pose'][1] - self.mfvs_last_goal_pose[1]) ** 2
                ) >= min_goal_separation
            )
        ]
        if not available_indices:
            self.get_logger().info(
                f"MFVS select: no available indices after filtering, remaining={len(self.mfvs_remaining_indices)}, "
                f"visited_keys={len(self.mfvs_visited_keys)}, last_goal={self.mfvs_last_goal_pose}"
            )
            return None

        scored_candidates = []
        for index in available_indices:
            candidate = self.mfvs_candidates[index]
            score = self._score_mfvs_candidate(candidate, robot_position, robot_heading)
            scored_candidates.append((index, score, candidate['pose']))

        best_index, best_score, best_pose = max(scored_candidates, key=lambda item: item[1])
        if not np.isfinite(best_score) or best_score == float('-inf'):
            self.get_logger().info(
                f"MFVS select: no finite score candidates, robot=({robot_position[0]:.2f}, {robot_position[1]:.2f}), "
                f"heading={robot_heading:.2f}, available={len(available_indices)}"
            )
            return None

        self.get_logger().info(
            f"MFVS select: remaining={len(self.mfvs_remaining_indices)}, available={len(available_indices)}, "
            f"robot=({robot_position[0]:.2f}, {robot_position[1]:.2f}), last_goal={self.mfvs_last_goal_pose}, "
            f"best_index={best_index}, best_pose=({best_pose[0]:.2f}, {best_pose[1]:.2f}), score={best_score:.4f}"
        )
        return best_index

    def _prune_mfvs_candidates(self, selected_candidate_index):
        selected_pose = self.mfvs_candidates[selected_candidate_index]['pose']
        prune_radius = max(self.exploration_points * self.mfvs_pruning_distance_factor, self.exploration_radius * self.mfvs_pruning_distance_factor, 1.0)

        self.mfvs_remaining_indices = [
            candidate_index
            for candidate_index in self.mfvs_remaining_indices
            if sqrt(
                (self.mfvs_candidates[candidate_index]['pose'][0] - selected_pose[0]) ** 2 +
                (self.mfvs_candidates[candidate_index]['pose'][1] - selected_pose[1]) ** 2
            ) >= prune_radius
        ]

    def _get_active_waypoint_pose(self, waypoint_index):
        if self.exploration_strategy == 'baseline':
            return self.waypoint_[waypoint_index]
        return self.mfvs_candidates[waypoint_index]['pose']

    def get_exploration_waypoints(self, target_name, offset_distance, additional_distance, robot_position):
        """Generate exploration waypoints for a target building."""
        target_ways = [way for way in self.osm_handler.ways_info 
                      if 'name' in way['tags'] and way['tags']['name'] == target_name]
        
        self.get_logger().debug(f"Target ways: {target_ways}")
        
        if not target_ways:
            self.get_logger().error(f"No building named '{target_name}' found.")
            return []
            
        target_way = target_ways[0]
        coordinates = self.osm_handler.get_way_nodes_locations(target_way['id'])

        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        transformer = CoordinateTransformer()
        utm_coordinates = []
        for lon, lat in coordinates:
            easting, northing, epsg = transformer.wgs84_to_utm(lon, lat)
            utm_coordinates.append((easting, northing))
        coordinates = [np.dot(self.transform_matrix, np.array([x, y, 1]))[:2] for x, y in utm_coordinates]
        
        inflated_building = self.inflate_building(coordinates, offset_distance)
        
        evenly_sampled_waypoints = self.sample_waypoints_evenly(inflated_building, additional_distance)
        
        center = np.mean(coordinates, axis=0)
        self.current_building_center = tuple(center)

        if self.exploration_strategy == 'baseline':
            orientations = self.calculate_orientations(evenly_sampled_waypoints, center)
            waypoints_with_yaw = [(x, y, yaw) for (x, y), yaw in zip(evenly_sampled_waypoints, orientations)]
            closest_index = self.get_closest_waypoint_index(waypoints_with_yaw, robot_position)
            return self.reorder_waypoints(waypoints_with_yaw, closest_index)

        self.mfvs_candidates = self._build_candidate_pool_mfvs(inflated_building, additional_distance)
        if not self.mfvs_candidates:
            self.get_logger().warn('MFVS candidate pool is empty, falling back to baseline sampling')
            orientations = self.calculate_orientations(evenly_sampled_waypoints, center)
            waypoints_with_yaw = [(x, y, yaw) for (x, y), yaw in zip(evenly_sampled_waypoints, orientations)]
            closest_index = self.get_closest_waypoint_index(waypoints_with_yaw, robot_position)
            return self.reorder_waypoints(waypoints_with_yaw, closest_index)

        self.mfvs_perimeter_length = max(candidate['arc_length'] for candidate in self.mfvs_candidates)
        self.mfvs_remaining_indices = list(range(len(self.mfvs_candidates)))
        self.mfvs_visited_positions = []
        self.mfvs_visited_arc_positions = []
        return [candidate['pose'] for candidate in self.mfvs_candidates]

    async def execute_callback(self, goal_handle):
        """Main execution callback for the action server"""
        self._current_goal_handle = goal_handle
        self._cancel_requested = False

        try:
            self.get_logger().info('Executing goal...')

            # wait for transform availability
            try:
                transform: TransformStamped = await self.tf_buffer.lookup_transform_async(
                    self.map_frame,
                    self.bk_frame,
                    rclpy.time.Time())
            except Exception as e:
                self.get_logger().error(f'Transform lookup failed: {str(e)}')
                goal_handle.abort()
                return EntranceExploration.Result(
                    success=False,
                    message=f'Transform lookup failed: {str(e)}'
                )
            
            self.cur_position = (transform.transform.translation.x, transform.transform.translation.y)
            self.get_logger().info(f'Current position: {self.cur_position}')
            
            # get target parameters
            self.target_building_id = goal_handle.request.building_id
            self.target_unit_id = goal_handle.request.unit_id
            
            # Parameter validation
            if not isinstance(self.target_building_id, str) or not isinstance(self.target_unit_id, str):
                goal_handle.abort()
                return EntranceExploration.Result(
                    success=False,
                    message="building_id and unit_id must be strings"
                )
            
            if not self.target_building_id or not self.target_unit_id:
                goal_handle.abort()
                return EntranceExploration.Result(
                    success=False,
                    message="building_id and unit_id cannot be empty"
                )

            self.waypoint_ = self.get_exploration_waypoints(
                self.target_building_id,
                self.exploration_radius,
                self.exploration_points,
                self.cur_position
            )
            
            if not self.waypoint_:
                goal_handle.abort()
                return EntranceExploration.Result(
                    success=False,
                    message="Failed to generate exploration waypoints"
                )

            self.remaining_waypoint_indices = list(range(len(self.waypoint_)))

            self.get_logger().debug(f"Generated waypoints: {self.waypoint_}")
            self.get_logger().info(f'Exploration strategy: {self.exploration_strategy}')
                
        except Exception as e:
            self.get_logger().error(f'Unexpected error: {str(e)}')
            goal_handle.abort()
            return EntranceExploration.Result(
                success=False,
                message=f'Exploration failed due to unexpected error: {str(e)}'
            )

        result = await self.exec_navigation(goal_handle)
        
        if self._cancel_requested:
            self.get_logger().info("Goal was canceled")
            goal_handle.canceled()
        elif result.success:
            self.get_logger().info("Goal completed successfully")
            goal_handle.succeed()
        else:
            self.get_logger().info("Goal failed")
            goal_handle.abort()

        self._reset_state()
        return result

    async def exec_navigation(self, goal_handle):
        """
        Execute navigation sequence through waypoints
        Args:
            goal_handle: The goal handle for the action
        Returns:
            EntranceExploration.Result: The result of the exploration
        """
        feedback_msg = EntranceExploration.Feedback()
        
        try:
            while (self.remaining_waypoint_indices if self.exploration_strategy == 'baseline' else self.mfvs_remaining_indices) and not self._cancel_requested:
                current_waypoint_index = await self._select_next_waypoint_index()
                if current_waypoint_index is None:
                    break

                if self.exploration_strategy != 'baseline':
                    current_candidate = self.mfvs_candidates[current_waypoint_index]
                    current_position, _ = await self._get_current_robot_state()
                    reference_position = current_position or self.cur_position
                    if reference_position is not None:
                        candidate_pose = current_candidate['pose']
                        near_duplicate_distance = max(0.75, 0.15 * max(self.exploration_points, self.exploration_radius, 1.0))
                        if sqrt((candidate_pose[0] - reference_position[0]) ** 2 + (candidate_pose[1] - reference_position[1]) ** 2) < near_duplicate_distance:
                            self.get_logger().info(
                                f"Skipping near-duplicate MFVS waypoint {current_waypoint_index}: ({candidate_pose[0]}, {candidate_pose[1]}), "
                                f"robot=({reference_position[0]:.2f}, {reference_position[1]:.2f}), last_goal={self.mfvs_last_goal_pose}, "
                                f"threshold={near_duplicate_distance:.2f}"
                            )
                            self._mfvs_discard_candidate(current_waypoint_index)
                            self.mfvs_visited_positions.append(candidate_pose[:2])
                            self.mfvs_visited_arc_positions.append(current_candidate['arc_length'])
                            continue

                self.current_waypoint_index = current_waypoint_index
                self.current_exploration_step += 1

                if self.exploration_strategy == 'baseline':
                    self.visited_waypoint_indices.add(current_waypoint_index)
                    self.remaining_waypoint_indices.remove(current_waypoint_index)
                else:
                    current_candidate = self.mfvs_candidates[current_waypoint_index]
                    self._mfvs_discard_candidate(current_waypoint_index)
                    self.mfvs_visited_positions.append(current_candidate['pose'][:2])
                    self.mfvs_visited_arc_positions.append(current_candidate['arc_length'])

                feedback_msg.status = (
                    f"Exploring waypoint {self.current_exploration_step}"
                )
                goal_handle.publish_feedback(feedback_msg)
                
                if await self._navigate_to_waypoint(goal_handle):
                    if await self._check_entrance(goal_handle):
                        self.is_task_success = True
                        return EntranceExploration.Result(
                            success=True,
                            message="Successfully found the target entrance"
                        )

            # Check if the task was canceled
            if self._cancel_requested:
                return EntranceExploration.Result(
                    success=False,
                    message="Task was canceled"
                )

            return EntranceExploration.Result(
                success=self.is_task_success,
                message="Completed exploration without finding target entrance"
            )
            
        except Exception as e:
            self.get_logger().error(f"Exception in exec_navigation: {str(e)}")
            return EntranceExploration.Result(
                success=False,
                message=f"Navigation failed: {str(e)}"
            )

    async def _navigate_to_waypoint(self, goal_handle):
        """
        Navigate to a single waypoint
        Returns:
            bool: True if navigation succeeded, False otherwise
        """
        waypoint = self._get_active_waypoint_pose(self.current_waypoint_index)
        self.mfvs_last_goal_pose = waypoint[:2]
        goal = self._create_pose_goal(waypoint)
        
        self.navigator.goToPose(goal)
        self.get_logger().info(f"Sent waypoint {self.current_exploration_step}: ({waypoint[0]}, {waypoint[1]})")
        
        return await self._wait_for_navigation(goal_handle)

    def euler_to_quaternion(self, yaw):
        """Convert a yaw angle to a quaternion."""
        q = Quaternion()
        q.w = cos(yaw / 2)
        q.x = 0.0
        q.y = 0.0
        q.z = sin(yaw / 2)
        return q

    def _create_pose_goal(self, waypoint):
        """Create a PoseStamped message from waypoint coordinates"""
        goal = PoseStamped()
        goal.header.frame_id = self.map_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = waypoint[0]
        goal.pose.position.y = waypoint[1]
        goal.pose.position.z = 0.0
        goal.pose.orientation = self.euler_to_quaternion(waypoint[2])
        return goal

    async def _wait_for_navigation(self, goal_handle):
        """
        Wait for the navigation task to complete and publish navigation progress
        Returns:
            bool: True if navigation succeeded, False otherwise
        """
        feedback_msg = EntranceExploration.Feedback()
        
        while not self.navigator.isTaskComplete():
            if self._cancel_requested:
                self.get_logger().info('Navigation task was cancelled')
                feedback_msg.status = "Navigation cancelled"
                goal_handle.publish_feedback(feedback_msg)
                return False
                
            # Get navigation feedback
            nav_feedback = self.navigator.getFeedback()
            if nav_feedback:
                feedback_msg.status = (
                    f"Navigating to waypoint {self.current_exploration_step}, "
                    f"Distance remaining: {nav_feedback.distance_remaining:.2f}m"
                )
                goal_handle.publish_feedback(feedback_msg)
                
                self.get_logger().info(feedback_msg.status)
            
            time.sleep(self.navigation_feedback_interval)

        # If the task is canceled, return False
        if self._cancel_requested:
            return False

        self.nav_result = self.navigator.getResult()
        if self.nav_result == TaskResult.SUCCEEDED:
            self.get_logger().info("Arrived at target waypoint")
            feedback_msg.status = "Reached waypoint, checking for entrance"
            goal_handle.publish_feedback(feedback_msg)
            return True
        else:
            self.get_logger().info(f'Navigation result: {self.nav_result}')
            feedback_msg.status = f"Navigation ended: {self.nav_result}"
            goal_handle.publish_feedback(feedback_msg)
            return False

    async def _check_entrance(self, goal_handle):
        """
        Check if the current location matches the target entrance
        Args:
            goal_handle: The goal handle for the action
        Returns:
            bool: True if entrance is found and matches target, False otherwise
        """
        feedback_msg = EntranceExploration.Feedback()
        
        if self.latest_image is None:
            self.get_logger().warn('No image data available')
            feedback_msg.status = "Waiting for camera image..."
            goal_handle.publish_feedback(feedback_msg)
            return False

        try:
            # Update status to show we're processing
            feedback_msg.status = "Processing image for entrance detection..."
            goal_handle.publish_feedback(feedback_msg)

            # Create service request
            request = GetEntranceId.Request()
            request.image = self.latest_image

            # Call entrance recognition service
            future = self.get_entrance_id_client.call_async(request)
            feedback_msg.status = "Calling entrance recognition service..."
            goal_handle.publish_feedback(feedback_msg)
            
            response = await future

            # Check if the detected entrance matches our target
            if response.success:
                detected_msg = f'Detected entrance: Unit {response.entrance_id}'
                self.get_logger().info(detected_msg)
                feedback_msg.status = detected_msg
                goal_handle.publish_feedback(feedback_msg)

                if f"unit{response.entrance_id}" == self.target_unit_id:
                    feedback_msg.status = "Target entrance found!"
                    goal_handle.publish_feedback(feedback_msg)
                    return True
                else:
                    feedback_msg.status = "Entrance detected but not the target one, continuing search..."
                    goal_handle.publish_feedback(feedback_msg)
                    return False
            else:
                self.get_logger().warn('No entrance detected in current image')
                feedback_msg.status = "No entrance detected in current view"
                goal_handle.publish_feedback(feedback_msg)
                return False

        except Exception as e:
            error_msg = f'Error during entrance check: {str(e)}'
            self.get_logger().error(error_msg)
            feedback_msg.status = f"Error in entrance detection: {str(e)}"
            goal_handle.publish_feedback(feedback_msg)
            return False

def main():
    rclpy.init()
    entrance_exploration_action_server = EntranceExplorationActionServer()

    executor = MultiThreadedExecutor()
    
    try:
        rclpy.spin(entrance_exploration_action_server, executor=executor)
    except KeyboardInterrupt:
        pass
    
    entrance_exploration_action_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()