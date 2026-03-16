import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from custom_interfaces.action import EntranceExploration
import signal
import sys
from rclpy.executors import MultiThreadedExecutor

class EntranceExplorationActionClient(Node):
    """Action client for entrance exploration."""

    def __init__(self):
        """Initialize the action client node."""
        super().__init__('entrance_exploration_action_client')
        self._action_client = ActionClient(
            self,
            EntranceExploration,
            'explore_entrance'
        )
        self._goal_handle = None
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals gracefully.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.get_logger().info('Received shutdown signal, canceling goal...')
        if self._goal_handle is not None:
            try:
                # Cancel the goal with timeout
                cancel_future = self._goal_handle.cancel_goal_async()
                # Wait for cancel response with timeout
                rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
                self.get_logger().info('Goal canceled')
            except Exception as e:
                self.get_logger().error(f'Error during goal cancellation: {str(e)}')
        
        # Force cleanup and exit
        try:
            self.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f'Error during shutdown: {str(e)}')
        finally:
            sys.exit(0)

    def cancel_goal(self):
        """Cancel the current goal with timeout mechanism."""
        if self._goal_handle is not None:
            self.get_logger().info('Canceling current goal...')
            try:
                # Send cancel request
                cancel_future = self._goal_handle.cancel_goal_async()
                # Wait for cancel response with timeout
                rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
                self.get_logger().info('Goal canceled')
            except Exception as e:
                self.get_logger().error(f'Error canceling goal: {str(e)}')
        else:
            self.get_logger().warn('No active goal to cancel')

    def send_goal(self, building_id: str, unit_id: str) -> None:
        """
        Send exploration goal to the action server.

        Args:
            building_id (str): Target building identifier
            unit_id (str): Target unit identifier
        """
        # Wait for action server to be available
        self._action_client.wait_for_server()
        
        # Create goal message
        goal_msg = EntranceExploration.Goal()
        goal_msg.building_id = building_id
        goal_msg.unit_id = unit_id
        
        # Send goal and set callbacks
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)
    
    # process goal acceptance/rejection response
    def goal_response_callback(self, future) -> None:
        """
        Callback for goal response.

        Args:
            future: Future object containing the goal response
        """
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected by server')
            return
            
        self.get_logger().info('Goal accepted by server')
        self._goal_handle = goal_handle
        
        # Request the result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future) -> None:
        """
        Callback for getting the result.

        Args:
            future: Future object containing the action result
        """
        try:
            result = future.result().result
            self.get_logger().info(f'Result: {result.success}')
            self.get_logger().info(f'Message: {result.message}')
        except Exception as e:
            self.get_logger().error(f'Error getting result: {str(e)}')
        finally:
            self._goal_handle = None
            rclpy.shutdown()
    
    def feedback_callback(self, feedback_msg) -> None:
        """
        Callback for receiving feedback.

        Args:
            feedback_msg: Feedback message from the action server
        """
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.status}')


def main():
    """Main function to run the action client."""
    try:
        # Initialize ROS client
        rclpy.init()
        
        # Create and run action client
        action_client = EntranceExplorationActionClient()
        action_client.send_goal('building1', '1')
        
        # Spin until shutdown
        rclpy.spin(action_client)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        # Cleanup
        if rclpy.ok():
            try:
                # Ensure the goal is canceled before exiting
                if hasattr(action_client, '_goal_handle') and action_client._goal_handle is not None:
                    action_client.cancel_goal()
                action_client.destroy_node()
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")
            finally:
                rclpy.shutdown()


if __name__ == '__main__':
    main()