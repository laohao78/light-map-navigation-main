import math
import sys
from pathlib import Path

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.action import ActionClient
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener
from ament_index_python.packages import get_package_share_directory

from custom_interfaces.action import EntranceExploration
from custom_interfaces.msg import TaskDescription
from custom_interfaces.srv import GetTaskSequence, TaskRecord, TriggerReplan
from utils_pkg import CoordinateTransformer, OSMHandler, OsmGlobalPlanner


class DeliveryClientNode(Node):
    def __init__(self):
        super().__init__('delivery_client_node')

        default_osm_file = str(
            Path(get_package_share_directory('utils_pkg')) / 'resource' / 'osm' / 'medium.osm'
        )
        self.declare_parameter('osm_file_path', default_osm_file)
        self.declare_parameter(
            'transform_matrix',
            [1.0, 0.0, 500000.0, 0.0, 1.0, 4483000.0, 0.0, 0.0, 1.0],
        )
        self.declare_parameter('osm_routing_url', 'http://10.219.235.175:5000/route/v1/driving/')

        self.navigator = BasicNavigator()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.task_planning_client = self.create_client(GetTaskSequence, 'task_planning')
        self.task_record_client = self.create_client(TaskRecord, 'task_record')
        self.replan_client = self.create_client(TriggerReplan, 'trigger_replan')
        self.exploration_action_client = ActionClient(self, EntranceExploration, 'explore_entrance')

        self.osm_handler = OSMHandler()
        osm_file = self.get_parameter('osm_file_path').value
        self.osm_handler.apply_file(osm_file)

    def _wait_for_service(self, client, service_name: str):
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'{service_name} service not available, waiting...')

    def _wait_for_action_server(self, client, server_name: str):
        while not client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info(f'{server_name} action server not available, waiting...')

    def _wait_for_replan_service(self):
        while not self.replan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Replan service not available, waiting...')

    def _get_task_sequence(self, task_description: str) -> list[TaskDescription]:
        self._wait_for_service(self.task_planning_client, 'Task planning')

        request = GetTaskSequence.Request()
        request.task_description = task_description

        future = self.task_planning_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response is None:
            raise RuntimeError('Task planning service returned no response')

        return list(response.task_sequence)

    def _send_task_record_request(self, status: str, address: str):
        self._wait_for_service(self.task_record_client, 'TaskRecord')

        request = TaskRecord.Request()
        request.status = status
        request.address = address

        future = self.task_record_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response is None:
            raise RuntimeError('TaskRecord service call failed, future did not return a result')

        if response.success:
            self.get_logger().info('TaskRecord service call successful.')
        else:
            self.get_logger().info('TaskRecord service call failed.')

    def _get_transform_matrix(self):
        flat_matrix = self.get_parameter('transform_matrix').value
        return np.array(flat_matrix).reshape(3, 3)

    def _get_osrm_url(self):
        return self.get_parameter('osm_routing_url').value

    def _get_robot_position(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
            )
            return (
                transform.transform.translation.x,
                transform.transform.translation.y,
            )
        except TransformException as ex:
            self.get_logger().error(f'Could not get robot position: {ex}')
            return None

    def _validate_robot_position(self, position) -> bool:
        if position is None:
            self.get_logger().error('Invalid robot position: None')
            return False
        if not isinstance(position, tuple) or len(position) != 2:
            self.get_logger().error('Invalid robot position format')
            return False
        if not all(isinstance(value, (int, float)) for value in position):
            self.get_logger().error('Invalid robot position coordinates')
            return False
        return True

    def _get_target_coordinates(self, task_info: str):
        info = task_info.lower().replace(' ', '')

        if ':' in info:
            building_id, unit_id = info.split(':', 1)
            unit_locations = self.osm_handler.get_node_location_by_name(unit_id)
            if not unit_locations:
                raise RuntimeError(f'Unit not found: {unit_id}')
            return unit_locations[0]

        building_centers = self.osm_handler.get_way_center_by_name(info)
        if not building_centers:
            raise RuntimeError(f'Building not found: {info}')

        return building_centers[0]

    def _yaw_to_quaternion(self, yaw: float):
        return (
            0.0,
            0.0,
            math.sin(yaw / 2.0),
            math.cos(yaw / 2.0),
        )

    def _create_navigation_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        quaternion = self._yaw_to_quaternion(yaw)
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        return pose

    def _target_to_map_pose(self, target_lon: float, target_lat: float) -> PoseStamped:
        utm_x, utm_y, _ = CoordinateTransformer.wgs84_to_utm(target_lon, target_lat)
        transform_matrix = self._get_transform_matrix()
        transform_matrix_inv = np.linalg.inv(transform_matrix)
        utm_coords = np.array([[utm_x], [utm_y], [1.0]])
        local_coords = np.dot(transform_matrix_inv, utm_coords)
        return self._create_navigation_pose(float(local_coords[0][0]), float(local_coords[1][0]), 0.0)

    def _robot_position_to_wgs84(self, robot_position, utm_epsg: int):
        transform_matrix = self._get_transform_matrix()
        position_vector = np.array([[robot_position[0]], [robot_position[1]], [1.0]])
        transformed_position = np.dot(transform_matrix, position_vector)

        utm_x = float(transformed_position[0][0])
        utm_y = float(transformed_position[1][0])

        return CoordinateTransformer.utm_to_wgs84(utm_x, utm_y, utm_epsg)

    def _route_waypoints_to_pose_list(self, waypoints):
        pose_list = []
        for index, waypoint in enumerate(waypoints):
            local_x, local_y = self._wgs84_to_local(waypoint.lon, waypoint.lat)

            yaw = 0.0
            if index < len(waypoints) - 1:
                next_local_x, next_local_y = self._wgs84_to_local(waypoints[index + 1].lon, waypoints[index + 1].lat)
                yaw = math.atan2(next_local_y - local_y, next_local_x - local_x)

            pose_list.append(self._create_navigation_pose(local_x, local_y, yaw))

        return pose_list

    def _pose_list_to_pose_array(self, pose_list):
        pose_array = PoseArray()
        pose_array.header.frame_id = 'map'
        pose_array.header.stamp = self.get_clock().now().to_msg()

        for pose_stamped in pose_list:
            pose = Pose()
            pose.position.x = pose_stamped.pose.position.x
            pose.position.y = pose_stamped.pose.position.y
            pose.position.z = pose_stamped.pose.position.z
            pose.orientation = pose_stamped.pose.orientation
            pose_array.poses.append(pose)

        return pose_array

    def _pose_array_to_pose_list(self, pose_array):
        pose_list = []
        for pose in pose_array.poses:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'map'
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.pose = pose
            pose_list.append(pose_stamped)

        return pose_list

    def _local_replan(self, pose_list):
        self._wait_for_replan_service()

        request = TriggerReplan.Request()
        request.waypoints = self._pose_list_to_pose_array(pose_list)

        future = self.replan_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response is None:
            raise RuntimeError('Replan service returned no response')

        if not response.success:
            self.get_logger().warn('Replanning failed, falling back to original route')
            return pose_list

        self.get_logger().info('Replanning successful')
        return self._pose_array_to_pose_list(response.new_waypoints)

    def _wgs84_to_local(self, lon: float, lat: float):
        utm_x, utm_y, _ = CoordinateTransformer.wgs84_to_utm(lon, lat)
        transform_matrix = self._get_transform_matrix()
        transform_matrix_inv = np.linalg.inv(transform_matrix)
        utm_coords = np.array([[utm_x], [utm_y], [1.0]])
        local_coords = np.dot(transform_matrix_inv, utm_coords)
        return float(local_coords[0][0]), float(local_coords[1][0])

    def _navigate_to_target(self, target_lon: float, target_lat: float) -> bool:
        robot_position = self._get_robot_position()
        if not self._validate_robot_position(robot_position):
            return False

        utm_epsg = CoordinateTransformer.get_utm_epsg(target_lon, target_lat)
        start_lon, start_lat = self._robot_position_to_wgs84(robot_position, utm_epsg)
        start_position = f'{start_lon:.9f},{start_lat:.9f}'
        target_position = f'{target_lon:.9f},{target_lat:.9f}'

        planner = OsmGlobalPlanner(self._get_osrm_url())
        route_waypoints = planner.get_route(start_position, target_position)
        if not route_waypoints:
            self.get_logger().error('Failed to generate route waypoints')
            return False

        pose_list = self._route_waypoints_to_pose_list(route_waypoints)
        if not pose_list:
            self.get_logger().error('Failed to convert route waypoints to poses')
            return False

        self.navigator.followWaypoints(pose_list)

        while not self.navigator.isTaskComplete():
            nav_feedback = self.navigator.getFeedback()
            if nav_feedback:
                current_waypoint = getattr(nav_feedback, 'current_waypoint', None)
                if current_waypoint is not None:
                    progress_text = f'{current_waypoint + 1}/{len(pose_list)}'
                else:
                    progress_text = 'in progress'
                self.get_logger().info(
                    f'Navigation in progress, remaining waypoints: {progress_text}'
                )

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            return True

        self.get_logger().warn(f'Navigation failed with result {result}, attempting replanning')
        replanned_pose_list = self._local_replan(pose_list)
        if not replanned_pose_list:
            return False

        self.navigator.followWaypoints(replanned_pose_list)
        while not self.navigator.isTaskComplete():
            nav_feedback = self.navigator.getFeedback()
            if nav_feedback:
                current_waypoint = getattr(nav_feedback, 'current_waypoint', None)
                if current_waypoint is not None:
                    progress_text = f'{current_waypoint + 1}/{len(replanned_pose_list)}'
                else:
                    progress_text = 'in progress'
                self.get_logger().info(
                    f'Replanned navigation in progress, remaining waypoints: {progress_text}'
                )

        return self.navigator.getResult() == TaskResult.SUCCEEDED

    def _execute_exploration(self, building_id: str, unit_id: str):
        self._wait_for_action_server(self.exploration_action_client, 'Exploration')

        goal_msg = EntranceExploration.Goal()
        goal_msg.building_id = building_id
        goal_msg.unit_id = unit_id

        send_goal_future = self.exploration_action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if goal_handle is None:
            raise RuntimeError('Failed to send exploration goal')
        if not goal_handle.accepted:
            return False, 'Exploration goal rejected'

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()
        if result is None:
            raise RuntimeError('Failed to get exploration result')

        exploration_result = result.result
        return exploration_result.success, exploration_result.message

    def _build_task_record_address(self, index: int, task: TaskDescription) -> str:
        return f'task_{index + 1}:{task.task_type}:{task.task_information}'

    def execute_delivery_tasks(self, user_input: str):
        task_sequence = self._get_task_sequence(user_input)

        if not task_sequence:
            raise RuntimeError('Task sequence is empty')

        total_tasks = len(task_sequence)
        self.get_logger().info(f'Received {total_tasks} planned tasks')

        for index, task in enumerate(task_sequence):
            record_address = self._build_task_record_address(index, task)

            self.get_logger().info(
                f'Starting task {index + 1}/{total_tasks}: {task.task_type} - {task.task_information}'
            )
            self._send_task_record_request('start', record_address)

            try:
                if task.task_type == 'NAVIGATION':
                    target_lon, target_lat = self._get_target_coordinates(task.task_information)
                    success = self._navigate_to_target(target_lon, target_lat)
                    message = 'Navigation succeeded' if success else 'Navigation failed'
                elif task.task_type == 'EXPLORATION':
                    building_id, unit_id = task.task_information.split(':', 1)
                    success, message = self._execute_exploration(building_id, unit_id)
                else:
                    raise RuntimeError(f'Unknown task type: {task.task_type}')

                if success:
                    self.get_logger().info(
                        f'Completed task {index + 1}/{total_tasks}: {task.task_type} - {message}'
                    )
                else:
                    raise RuntimeError(f'Delivery task failed: {message}')
            finally:
                self._send_task_record_request('end', record_address)

        self.get_logger().info('All delivery tasks completed successfully')


def main():
    rclpy.init()
    node = DeliveryClientNode()

    try:
        if len(sys.argv) > 1:
            user_input = ' '.join(sys.argv[1:]).strip()
        else:
            user_input = input('\nPlease enter delivery instructions:\n').strip()

        if not user_input:
            raise ValueError('Empty delivery instruction')

        node.execute_delivery_tasks(user_input)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error occurred: {str(e)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()