import rclpy
from rclpy.node import Node
from custom_interfaces.srv import GetTaskSequence
from custom_interfaces.msg import TaskDescription

class TaskPlanningClient(Node):
    """
    ROS2 client node for task planning service.
    Sends task descriptions and receives planned task sequences.
    """
    
    def __init__(self):
        """Initialize the task planning client node"""
        super().__init__('task_planning_client')
        
        # Create service client
        self._init_client()
        
        # Wait for service availability
        self._wait_for_service()
        
    def _init_client(self):
        """Initialize the service client"""
        self.client = self.create_client(
            GetTaskSequence,
            'task_planning'
        )
    
    def _wait_for_service(self):
        """Wait for the service to become available"""
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
    
    def send_request(self, task_description: str) -> rclpy.task.Future:
        """
        Send a task planning request
        
        Args:
            task_description: Natural language description of the delivery task
            
        Returns:
            Future object for the service response
        """
        request = GetTaskSequence.Request()
        request.task_description = task_description
        return self.client.call_async(request)
    
    def process_response(self, future: rclpy.task.Future):
        """
        Process and log the service response
        
        Args:
            future: Future object containing the service response
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            response = future.result()
            if response is not None:
                self._log_task_sequence(response.task_sequence)
                return True
            else:
                self.get_logger().error('Service call failed')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Error processing response: {str(e)}')
            return False
    
    def _log_task_sequence(self, task_sequence):
        """
        Log the received task sequence
        
        Args:
            task_sequence: List of TaskDescription messages
        """
        self.get_logger().info('Received task sequence:')
        for i, task in enumerate(task_sequence):
            self.get_logger().info(f'Task {i+1}:')
            self.get_logger().info(f'  Type: {task.task_type}')
            self.get_logger().info(f'  Description: {task.task_information}')

def main(args=None):
    """Main function to run the client node"""
    rclpy.init(args=args)
    client = TaskPlanningClient()
    
    try:
        # Example task description
        test_task = "Please deliver this package to unit 1 of building 1 and this apple to unit 2 of building 1."
        
        # Send request and wait for response
        future = client.send_request(test_task)
        rclpy.spin_until_future_complete(client, future)
        
        # Process response
        if not client.process_response(future):
            client.get_logger().error('Failed to process task sequence')
            
    except KeyboardInterrupt:
        client.get_logger().info('Client interrupted by user')
    except Exception as e:
        client.get_logger().error(f'Unexpected error: {str(e)}')
    finally:
        # Cleanup
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 