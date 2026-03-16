import rclpy
from rclpy.node import Node
from custom_interfaces.srv import GetEntranceId
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rclpy.parameter import Parameter
import argparse

class EntranceRecognitionClient(Node):
    """
    Client node for entrance recognition service.
    Handles image processing and service requests for entrance identification.
    """

    def __init__(self):
        super().__init__('entrance_recognition_client')
        
        # Declare parameters
        self.declare_parameter('image_path', '')
        
        self._init_client()
        self.bridge = CvBridge()

    def _init_client(self):
        """Initialize service client and wait for service availability"""
        self.client = self.create_client(GetEntranceId, 'entrance_recognition')
        self._wait_for_service()

    def _wait_for_service(self):
        """Wait for the recognition service to become available"""
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

    def send_request(self, image_path):
        """
        Send entrance recognition request with image to service
        
        Args:
            image_path (str): Path to the image file to be processed
            
        Returns:
            None: Results are logged to console
        """
        try:
            # Process image and create request
            ros_image = self._prepare_image(image_path)
            if ros_image is None:
                return

            future = self._send_service_request(ros_image)
            self._handle_service_response(future)

        except Exception as e:
            self.get_logger().error(f'Error sending request: {str(e)}')

    def _prepare_image(self, image_path):
        """
        Read and convert image to ROS format
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Image: ROS Image message or None if processing fails
        """
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            self.get_logger().error(f'Failed to read image: {image_path}')
            return None
        
        return self.bridge.cv2_to_imgmsg(cv_image, "bgr8")

    def _send_service_request(self, ros_image):
        """
        Send request to recognition service
        
        Args:
            ros_image (Image): ROS Image message to be processed
            
        Returns:
            Future: Service call future object
        """
        request = GetEntranceId.Request()
        request.image = ros_image
        return self.client.call_async(request)

    def _handle_service_response(self, future):
        """
        Process and log service response
        
        Args:
            future: Service call future object
        """
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            self.get_logger().info(
                f'Service response - Success: {response.success}, '
                f'Entrance ID: {response.entrance_id}, '
                f'Message: {response.message}'
            )
        else:
            self.get_logger().error('Service call failed')

    def get_image_path(self):
        """Get image path from parameter"""
        return self.get_parameter('image_path').value


def main(args=None):
    """Main function to initialize and run the client node"""
    # Initialize ROS2 with original args
    rclpy.init(args=args)
    
    # Create client node
    client = EntranceRecognitionClient()
    
    # Get image path from ROS2 parameter
    image_path = client.get_image_path()
    if not image_path:
        client.get_logger().error('No image path provided. Please use: ros2 run building_entrance_recognition entrance_recognition_client --ros-args -p image_path:=/path/to/image.png')
        client.destroy_node()
        rclpy.shutdown()
        return

    client.get_logger().info(f'Processing image: {image_path}')
    client.send_request(image_path)
    
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main() 