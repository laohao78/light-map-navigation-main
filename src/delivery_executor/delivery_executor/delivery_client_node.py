import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from action_msgs.msg import GoalStatus

from custom_interfaces.action import DeliveryTask
from custom_interfaces.msg import TaskDescription
from custom_interfaces.srv import GetTaskSequence, TaskRecord


class DeliveryClientNode(Node):
    def __init__(self):
        super().__init__('delivery_client_node')

        self.delivery_action_client = ActionClient(self, DeliveryTask, 'execute_delivery')
        self.task_planning_client = self.create_client(GetTaskSequence, 'task_planning')
        self.task_record_client = self.create_client(TaskRecord, 'task_record')

    def _wait_for_service(self, client, service_name: str):
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'{service_name} service not available, waiting...')

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

    def _wait_for_goal_result(self, goal_handle):
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result()

    def _send_delivery_goal(self, task_description: str):
        while not self.delivery_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Delivery action server not available, waiting...')

        goal_msg = DeliveryTask.Goal()
        goal_msg.user_input = task_description

        send_goal_future = self.delivery_action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if goal_handle is None:
            raise RuntimeError('Failed to send delivery goal')

        if not goal_handle.accepted:
            return False, 'Task rejected'

        result = self._wait_for_goal_result(goal_handle)
        if result is None:
            raise RuntimeError('Failed to get delivery result')

        status = result.status
        action_result = result.result
        if hasattr(action_result, 'result') and hasattr(action_result.result, 'message'):
            message = action_result.result.message
        elif hasattr(action_result, 'message'):
            message = action_result.message
        else:
            message = str(action_result)

        if status == GoalStatus.STATUS_SUCCEEDED:
            return True, message
        if status == GoalStatus.STATUS_CANCELED:
            return False, message
        return False, message

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
                success, message = self._send_delivery_goal(task.task_information)
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