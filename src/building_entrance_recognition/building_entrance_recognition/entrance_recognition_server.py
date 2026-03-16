import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
from custom_interfaces.srv import GetEntranceId
import base64
import json
from utils_pkg import APIClient
import yaml
import os
import re

class EntranceRecognitionService(Node):
    SYSTEM_PROMPT = """Now I need your help to analysis the picture as my assistant. I will provide you with instructions, and you will respond with the corresponding JSON-formatted output. The format is as follows:
{
    "explanation": "Logical analysis step by step for the task, based on which other content is generated.",
    "unit_number": "Only the unit number." # If you cannot find the unit number, please return an empty string.
}
For example:
{
    "explanation": "The picture shows Unit 2.",
    "unit_number": "2"
}
Note: It is necessary to output the result in JSON format and nothing else."""

    def __init__(self):
        super().__init__('entrance_recognition_server')
        self.srv = self.create_service(GetEntranceId, 'entrance_recognition', self.recognize_entrance_callback)
        self.bridge = CvBridge()
        self._init_api_client()

    def _init_api_client(self):
        """Initialize API client with configuration from yaml file"""
        try:
            # Get the package path
            package_path = os.path.dirname(os.path.dirname(__file__))
            config_path = os.path.join(package_path, 'config', 'api_config.yaml')
            
            # Load configuration
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Initialize API client
            self.client = APIClient(
                api_key=config['vlm_api']['key'],
                base_url=config['vlm_api']['base_url'],
                model_name=config['vlm_api']['model_name']
            )
        except Exception as e:
            self.get_logger().error(f'Failed to load API configuration: {str(e)}')
            raise

    def recognize_entrance_callback(self, request, response):
        self.get_logger().info('Received entrance recognition request')

        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(request.image, "bgr8")
            
            # Process the image and get entrance ID
            entrance_id = self.recognize_entrance(cv_image)
            
            if entrance_id:
                response.entrance_id = entrance_id
                response.success = True
                response.message = "Successfully detected entrance"
            else:
                response.entrance_id = ""
                response.success = False
                response.message = "No entrance detected"

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
            response.entrance_id = ""
            response.success = False
            response.message = f"Error: {str(e)}"

        return response

    def recognize_entrance(self, image):
        """
        Recognize entrance unit number from image using LLM API
        
        Args:
            image (np.ndarray): OpenCV image in BGR format
            
        Returns:
            str: Unit number if detected, None otherwise
        """
        try:
            # Convert image to base64 format
            image_base64 = self._convert_image_to_base64(image)
            
            # Get LLM response
            response = self._get_llm_response(image_base64)
            
            # Parse and validate response
            return self._parse_llm_response(response)
            
        except Exception as e:
            self.get_logger().error(f"Error in entrance recognition: {str(e)}")
            return None

    def _convert_image_to_base64(self, image):
        """Convert OpenCV image to base64 string"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def _get_llm_response(self, image_base64):
        """Get response from LLM API"""
        return self.client.chat(
            prompt="Please analyze this image and identify the unit number.",
            system_prompt=self.SYSTEM_PROMPT,
            image_data=image_base64
        )

    def _parse_llm_response(self, response):
        """Parse and validate LLM response"""
        if response and response[0]:
            try:
                pattern = r'\{[^{}]*\}'
                matches = re.findall(pattern, response[0])
                if matches:
                    json_str = matches[0]
                    result = json.loads(json_str)

                    return result.get("unit_number")
                else:
                    self.get_logger().error("No JSON object found in response")
                    self.get_logger().error(f"Raw response: {response[0]}")
                    
            except json.JSONDecodeError as e:
                self.get_logger().error(f"Failed to parse LLM response as JSON: {e}")
                self.get_logger().error(f"Raw response: {response[0]}")
        return None

def main(args=None):
    rclpy.init(args=args)
    node = EntranceRecognitionService()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
