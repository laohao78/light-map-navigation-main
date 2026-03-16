import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from custom_interfaces.action import DeliveryTask
from rclpy.callback_groups import ReentrantCallbackGroup
import signal
import sys

class DeliveryExecutorActionClient(Node):
    """Action client node for initiating and monitoring delivery tasks."""

    def __init__(self):
        super().__init__('delivery_executor_action_client')
        self._is_finished = False
        self._goal_handle = None
        
        callback_group = ReentrantCallbackGroup()
        
        self._action_client = ActionClient(
            self,
            DeliveryTask,
            'execute_delivery',
            callback_group=callback_group
        )
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.get_logger().info('Delivery Executor Action Client started')
        self._send_goal_future = None
        self._get_result_future = None

    def _signal_handler(self, signum, frame):
        self.get_logger().info('Received shutdown signal, canceling task...')
        try:
            if self._goal_handle is not None:
                cancel_future = self._goal_handle.cancel_goal_async()
                if not rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=1.0):
                    self.get_logger().warn('Cancel goal timed out, forcing exit...')
            
            self.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f'Error during shutdown: {str(e)}')
        finally:
            sys.exit(1)

    def cancel_goal(self):
        """Cancel current task with timeout mechanism"""
        if self._goal_handle is not None:
            self.get_logger().info('Canceling current task...')
            try:
                cancel_future = self._goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
                self.get_logger().info('Task canceled')
            except Exception as e:
                self.get_logger().error(f'Error canceling task: {str(e)}')
        else:
            self.get_logger().warn('No active task to cancel')

    def send_goal(self, user_input):
        """
        Send a delivery task goal to the action server.
        
        Args:
            user_input (str): The delivery task details provided by the user
            
        Returns:
            bool: False if input validation fails
        """
        self.get_logger().info('Waiting for Action Server...')
        
        while not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Action Server not available, waiting...')
        
        if not user_input or not isinstance(user_input, str):
            self.get_logger().error('Invalid user input')
            return False
        
        goal_msg = DeliveryTask.Goal()
        goal_msg.user_input = user_input
        
        self.get_logger().info(f'Sending delivery task goal: {user_input}')
        
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Handle the response to our goal request.
        
        Args:
            future: Future object containing the goal response
        """
        try:
            goal_handle = future.result()
            
            if not goal_handle.accepted:
                self.get_logger().info('Task rejected')
                self._is_finished = True
                return
            
            self.get_logger().info('Task accepted')
            self._goal_handle = goal_handle
            
            self._get_result_future = goal_handle.get_result_async()
            self._get_result_future.add_done_callback(self.get_result_callback)
            
        except Exception as e:
            self.get_logger().error(f'Error handling task response: {str(e)}')
            self._is_finished = True

    def get_result_callback(self, future):
        """
        Process the final result of the delivery task.
        
        Args:
            future: Future object containing the task result
        """
        status = future.result().status
        result = future.result().result
        result_message = result.result.message
        
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Task completed successfully: {result_message}')
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info(f'Task canceled: {result_message}')
        else:
            self.get_logger().info(f'Task failed: {result_message}')
        
        self.get_logger().info('Task processing complete, preparing to exit...')
        self._goal_handle = None
        self._is_finished = True
        
        self.destroy_node()
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        """
        Handle feedback updates from the action server.
        
        Args:
            feedback_msg: Contains the DeliveryFeedback message with progress information
        """
        status = feedback_msg.feedback.feedback.status
        self.get_logger().info(f'Received feedback:\n{status}')

    def destroy_node(self):
        """
        Clean up resources before node shutdown.
        """
        if self._action_client:
            self._action_client.destroy()
        super().destroy_node()

def main():
    """Main function to run the action client."""
    try:
        rclpy.init()
        
        action_client = DeliveryExecutorActionClient()
        action_client.send_goal("Please deliver this apple to unit 1 of building 1")
        
        rclpy.spin(action_client)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        if rclpy.ok():
            try:
                if hasattr(action_client, '_goal_handle') and action_client._goal_handle is not None:
                    action_client.cancel_goal()
                action_client.destroy_node()
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")
            finally:
                rclpy.shutdown()

if __name__ == '__main__':
    main() 