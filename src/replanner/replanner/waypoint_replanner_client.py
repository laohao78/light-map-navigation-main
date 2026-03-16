import rclpy
from rclpy.node import Node
from custom_interfaces.srv import TriggerReplan
from geometry_msgs.msg import PoseArray, Pose
import numpy as np

class WaypointReplannerClient(Node):
    def __init__(self):
        super().__init__('waypoint_replanner_client')
        
        # Create service client
        self.client = self.create_client(TriggerReplan, 'trigger_replan')
        
        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service...')
            
        self.get_logger().info('Service connected')
        
        test_waypoints = self.generate_test_waypoints()
        future = self.send_replan_request(test_waypoints)
        future.add_done_callback(self.replan_response_callback)
        
    def send_replan_request(self, waypoints):
        """Send replanning request"""
        request = TriggerReplan.Request()
        request.waypoints = waypoints
        return self.client.call_async(request)
        
    def replan_response_callback(self, future):
        """Handle replanning response"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Replanning successful')
                # Update print statement to match Pose type
                for i, pose in enumerate(response.new_waypoints.poses):
                    self.get_logger().info(
                        f'Waypoint {i}: x={pose.position.x:.2f}, y={pose.position.y:.2f}'
                    )
            else:
                self.get_logger().warn('Replanning failed')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

    def generate_test_waypoints(self):
        """Generate test waypoints"""
        waypoints = PoseArray()
        waypoints.header.frame_id = 'map'
        waypoints.header.stamp = self.get_clock().now().to_msg()
        
        # Test waypoint coordinates
        points = [
            (0.0, 5.0),
            (4.0, 5.0),
            (8.0, 5.0),
            (12.0, 5.0),
            (16.0, 5.0)
        ]
        
        for x, y in points:
            self.get_logger().info(f'Generating waypoint: x={x:.2f}, y={y:.2f}')
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.0
            # Set quaternion for forward orientation
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0
            waypoints.poses.append(pose)
        
        return waypoints

def main(args=None):
    rclpy.init(args=args)
    client = WaypointReplannerClient()
    rclpy.spin(client)
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()