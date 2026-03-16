import torch
import numpy as np
import cv2
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image_numpy, predict
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import RegionOfInterest, Image
from cv_bridge import CvBridge, CvBridgeError
import time
from typing import Tuple, List
from custom_interfaces.msg import GroundedSam

class GS2_ROS_Wrapper(Node):
    """ROS2 wrapper for Grounded SAM 2 model."""

    def __init__(self):
        """Initialize the Grounded SAM 2 ROS wrapper."""
        super().__init__('grounded_sam2_node')
        
        # Declare all ROS2 parameters
        self.declare_parameter('gsam2.scale', 1.0)
        self.declare_parameter('gsam2.do_sharpen', False)
        self.declare_parameter('gsam2.base_path', '/workspaces/light-map-navigation/src/grounded_sam2/grounded_sam2/Grounded-SAM-2/')
        self.declare_parameter('gsam2.text_prompt', 'car . house . road')
        self.declare_parameter('gsam2.sam2_checkpoint', './checkpoints/sam2.1_hiera_small.pt')
        self.declare_parameter('gsam2.sam2_model_config', 'configs/sam2.1/sam2.1_hiera_s.yaml')
        self.declare_parameter('gsam2.grounding_dino_config', 'grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py')
        self.declare_parameter('gsam2.grounding_dino_checkpoint', 'gdino_checkpoints/groundingdino_swint_ogc.pth')
        self.declare_parameter('gsam2.box_threshold', 0.3)
        self.declare_parameter('gsam2.text_threshold', 0.25)
        self.declare_parameter('gsam2.area_threshold', 80)
        self.declare_parameter('gsam2.processing_rate', 10.0)
        
        # Initialize model parameters
        self.get_logger().info("Initializing Grounded SAM 2 ROS Wrapper")
        self._setup_model_parameters()
        self._initialize_models()
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Initialize image buffer
        self._latest_image = None
        self._latest_header = None
        self._processing_lock = False
        
        # Setup ROS2 communication
        self._setup_ros_communication()
        
        # Setup timer for processing
        processing_rate = self.get_parameter('gsam2.processing_rate').get_parameter_value().double_value
        self.timer = self.create_timer(1.0/processing_rate, self.timer_callback)
        
        self.get_logger().info("Ready to receive images from topic")

    def _setup_model_parameters(self) -> None:
        """Setup model parameters and paths."""
        # Get base path and append to model paths
        self.BASE_PATH = self.get_parameter('gsam2.base_path').get_parameter_value().string_value
        
        # Get all parameters
        self.TEXT_PROMPT = self.get_parameter('gsam2.text_prompt').get_parameter_value().string_value
        self.SAM2_CHECKPOINT = self.BASE_PATH + self.get_parameter('gsam2.sam2_checkpoint').get_parameter_value().string_value
        self.SAM2_MODEL_CONFIG = self.get_parameter('gsam2.sam2_model_config').get_parameter_value().string_value
        self.GROUNDING_DINO_CONFIG = self.BASE_PATH + self.get_parameter('gsam2.grounding_dino_config').get_parameter_value().string_value
        self.GROUNDING_DINO_CHECKPOINT = self.BASE_PATH + self.get_parameter('gsam2.grounding_dino_checkpoint').get_parameter_value().string_value
        
        # Get threshold parameters
        self.BOX_THRESHOLD = self.get_parameter('gsam2.box_threshold').get_parameter_value().double_value
        self.TEXT_THRESHOLD = self.get_parameter('gsam2.text_threshold').get_parameter_value().double_value
        self.AREA_THRESHOLD = self.get_parameter('gsam2.area_threshold').get_parameter_value().integer_value
        
        # Set device
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def _initialize_models(self) -> None:
        """Initialize SAM2 and Grounding DINO models."""
        # Build SAM2 image predictor
        sam2_model = build_sam2(self.SAM2_MODEL_CONFIG, self.SAM2_CHECKPOINT, device=self.DEVICE)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # Build grounding dino model
        self.grounding_model = load_model(
            model_config_path=self.GROUNDING_DINO_CONFIG,
            model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT,
            device=self.DEVICE
        )

    def _setup_ros_communication(self) -> None:
        """Setup ROS2 subscribers and publishers."""
        self.subscription = self.create_subscription(
            Image,
            '/camera_sensor/image_raw',
            self.image_callback,
            10
        )
        
        self.detection_pub = self.create_publisher(
            GroundedSam,
            'grounded_sam2/detections',
            10
        )
        
        self.vis_pub = self.create_publisher(
            Image,
            'grounded_sam2/visualization',
            10
        )

        # Add publisher for binary mask image
        self.mask_pub = self.create_publisher(
            Image,
            'grounded_sam2/mask',
            10
        )

    def image_callback(self, msg: Image) -> None:
        """
        Store the latest image in buffer.
        
        Args:
            msg (Image): Input image message from ROS topic
        """
        self._latest_image = msg
        self._latest_header = msg.header

    def timer_callback(self) -> None:
        """Process the latest image at a fixed rate."""
        # Skip if no image available or processing is locked
        if self._latest_image is None or self._processing_lock:
            return
            
        self._processing_lock = True
        
        try:
            # Get the latest image
            image = self.bridge.imgmsg_to_cv2(self._latest_image, "rgb8")
            header = self._latest_header
            
            # Clear the buffer after getting the image
            self._latest_image = None
            self._latest_header = None
            
            # Process the image
            self.inference(image, header)
            
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
        finally:
            self._processing_lock = False

    def preprocess_image(self, image: np.ndarray, scale: float, do_sharpen: bool) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess image with resizing and optional sharpening."""
        orig_shape = (image.shape[1], image.shape[0])
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        image = cv2.resize(image, (new_width, new_height))
        
        if do_sharpen:
            blurred = cv2.GaussianBlur(image, (9, 9), 10)
            image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
            
        return image, orig_shape

    def inference_model_intern(self, np_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str], List[float]]:
        """Run inference with both Grounding DINO and SAM2 models."""
        image_source, image = load_image_numpy(np_img)
        self.sam2_predictor.set_image(image_source)

        # Grounding DINO inference
        with torch.autocast(device_type="cuda", enabled=False):
            boxes, confidences, labels = predict(
                model=self.grounding_model,
                image=image,
                caption=self.TEXT_PROMPT,
                box_threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_THRESHOLD,
            )
        
        self.get_logger().debug(f"Initial detection - boxes: {len(boxes)}, labels: {labels}")
        
        # Convert boxes to correct format
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        if len(input_boxes) == 0:
            self.get_logger().warn("No boxes detected")
            return [], [], [], []
        
        # SAM2 inference
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            masks, _, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        return input_boxes, masks, labels, confidences.numpy().tolist()

    def inference(self, image: np.ndarray, rgb_header) -> None:
        """Main inference pipeline."""
        timings = {}

        # Get parameters and preprocess image
        scale = self.get_parameter('gsam2.scale').get_parameter_value().double_value
        do_sharpen = self.get_parameter('gsam2.do_sharpen').get_parameter_value().bool_value
        
        preprocess_start = time.time()
        image, orig_shape = self.preprocess_image(image, scale, do_sharpen)
        timings['preprocess'] = time.time() - preprocess_start

        # Model inference
        model_start = time.time()
        boxes, masks, class_names, confidences = self.inference_model_intern(image)
        timings['model'] = time.time() - model_start

        # Post-processing
        post_start = time.time()
        height, width = image.shape[:2]
        label_image = np.full((height, width), -1, np.int16)
        bboxes_ros = []
        filtered_class_names = []
        filtered_confidences = []

        # Process detections
        for idx, (box, mask, class_name, confidence) in enumerate(zip(boxes, masks, class_names, confidences)):
            bb = RegionOfInterest()
            if box.ndim > 1:
                box = box[0]
            
            # Scale back coordinates
            xmin, ymin = int(box[0]/scale), int(box[1]/scale)
            xmax, ymax = int(box[2]/scale), int(box[3]/scale)
            
            # Create ROI message
            bb.x_offset, bb.y_offset = xmin, ymin
            bb.height = ymax - ymin
            bb.width = xmax - xmin
            bb.do_rectify = False
            
            # Filter small detections
            if (bb.width * bb.height) < self.AREA_THRESHOLD:
                continue
            
            bboxes_ros.append(bb)
            label_image[mask > 0] = idx
            filtered_class_names.append(class_name)
            filtered_confidences.append(confidence)

        timings['post_process'] = time.time() - post_start

        # Publish results
        publish_start = time.time()
        if filtered_class_names:
            self._publish_results(label_image, orig_shape, rgb_header, image, 
                               bboxes_ros, filtered_class_names, filtered_confidences)
        timings['publish'] = time.time() - publish_start

        # Log timing statistics
        self._log_statistics(timings, filtered_class_names)

    def _publish_results(self, label_image: np.ndarray, orig_shape: Tuple[int, int], 
                        rgb_header, image: np.ndarray, bboxes_ros: List[RegionOfInterest], 
                        class_names: List[str], confidences: List[float]) -> None:
        """Publish detection results using GroundedSam message."""
        detection_msg = GroundedSam()
        detection_msg.header = rgb_header
        detection_msg.source_image = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
        detection_msg.source_image.header = rgb_header
        detection_msg.label_image = self.bridge.cv2_to_imgmsg(label_image, encoding="16SC1")
        detection_msg.label_image.header = rgb_header
        detection_msg.class_names = class_names
        detection_msg.confidences = confidences
        detection_msg.boxes = bboxes_ros
        
        self.detection_pub.publish(detection_msg)
        
        vis_image = self._create_visualization(image, label_image, bboxes_ros, 
                                             class_names, confidences)
        vis_msg = self.bridge.cv2_to_imgmsg(vis_image, encoding="rgb8")
        vis_msg.header = rgb_header
        self.vis_pub.publish(vis_msg)

        # Create and publish binary mask
        binary_mask = self._create_binary_mask(label_image)
        mask_msg = self.bridge.cv2_to_imgmsg(binary_mask, encoding="mono8")
        mask_msg.header = rgb_header
        self.mask_pub.publish(mask_msg)

    def _create_visualization(self, image: np.ndarray, label_image: np.ndarray, 
                            bboxes: List[RegionOfInterest], class_names: List[str], 
                            confidences: List[float]) -> np.ndarray:
        """Create visualization image with bounding boxes and labels."""
        vis_image = image.copy()
        for idx, (bb, class_name, confidence) in enumerate(zip(bboxes, class_names, confidences)):
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(vis_image, 
                         (bb.x_offset, bb.y_offset), 
                         (bb.x_offset + bb.width, bb.y_offset + bb.height), 
                         color, 2)
            
            # Add label and confidence
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis_image, label, 
                       (bb.x_offset, bb.y_offset - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add semi-transparent mask overlay
            mask = (label_image == idx)
            vis_image[mask] = vis_image[mask] * 0.7 + np.array([0, 255, 0], dtype=np.uint8) * 0.3
        
        return vis_image

    def _create_binary_mask(self, label_image: np.ndarray) -> np.ndarray:
        """Create a binary mask where any detection is marked as white (255)."""
        binary_mask = np.zeros_like(label_image, dtype=np.uint8)
        binary_mask[label_image >= 0] = 255
        return binary_mask

    def _log_statistics(self, timings: dict, class_names: List[str]) -> None:
        """Log timing statistics and detection results."""
        self.get_logger().info("Time statistics:")
        for key, value in timings.items():
            self.get_logger().info(f"- {key.capitalize()}: {value:.3f}s")
        self.get_logger().info(f"Total processing: {sum(timings.values()):.3f}s")
        self.get_logger().info(f"Objects detected: {len(class_names)} ({class_names})")


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    gs2_node = GS2_ROS_Wrapper()
    
    try:
        rclpy.spin(gs2_node)
    except KeyboardInterrupt:
        pass
    finally:
        gs2_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
