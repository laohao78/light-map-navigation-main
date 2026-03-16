#! /opt/conda/envs/dl_env/bin/python

from flask import Flask, request, jsonify
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
import base64
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

app = Flask(__name__)

class Config:
    """Configuration class for model parameters and paths"""
    TEXT_PROMPT = "car. tire."
    IMG_PATH = "notebooks/images/truck.jpg"
    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo")
    DUMP_JSON_RESULTS = False

# Initialize models and create output directory
Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize models
sam2_model = build_sam2(Config.SAM2_MODEL_CONFIG, Config.SAM2_CHECKPOINT, device=Config.DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
grounding_model = load_model(
    model_config_path=Config.GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=Config.GROUNDING_DINO_CHECKPOINT,
    device=Config.DEVICE
)

class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    def decode_base64_image(img_base64: str) -> np.ndarray:
        """
        Decode base64 image string to numpy array
        
        Args:
            img_base64: Base64 encoded image string
            
        Returns:
            numpy.ndarray: Decoded image array
        """
        img_data = base64.b64decode(img_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """
        Encode image file to base64 string
        
        Args:
            image_path: Path to image file
            
        Returns:
            str: Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    @staticmethod
    def single_mask_to_rle(mask: np.ndarray) -> dict:
        """
        Convert single mask to RLE format
        
        Args:
            mask: Binary mask array
            
        Returns:
            dict: RLE encoded mask
        """
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

@app.route('/process_image', methods=['POST'])
def process_image():
    """
    Process image endpoint that handles image segmentation requests
    
    Returns:
        JSON response containing labels and masks
    """
    try:
        # Get request data
        data = request.json
        img_base64 = data.get('img_base64')
        if not img_base64:
            return jsonify({"error": "No image data provided"}), 400
            
        text_prompt = data.get('text_prompt', Config.TEXT_PROMPT)
        
        # Process image
        image_source = ImageProcessor.decode_base64_image(img_base64)
        temp_img_path = os.path.join(Config.OUTPUT_DIR, "temp_input.jpg")
        cv2.imwrite(temp_img_path, image_source)

        # Process image with models
        _, image = load_image(temp_img_path)
        sam2_predictor.set_image(image_source)

        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=text_prompt,
            box_threshold=Config.BOX_THRESHOLD,
            text_threshold=Config.TEXT_THRESHOLD,
        )

        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # FIXME: figure how does this influence the G-DINO model
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = confidences.numpy().tolist()
        class_names = labels

        class_ids = np.array(list(range(len(class_names))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]

        """
        Visualize image with supervision useful API
        """
        img = cv2.imread(temp_img_path)
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(os.path.join(Config.OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(Config.OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

        output_image_path = os.path.join(Config.OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg")
        cv2.imwrite(output_image_path, annotated_frame)

        encoded_string = ""
        try:
            with open(output_image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return jsonify({"error": "Failed to encode image"}), 500

        """
        Dump the results in standard format and save as json files
        """

        if Config.DUMP_JSON_RESULTS:
            # convert mask into rle format
            mask_rles = [ImageProcessor.single_mask_to_rle(mask) for mask in masks]

            input_boxes = input_boxes.tolist()
            scores = scores.tolist()
            # save the results in standard format
            results = {
                "image_path": temp_img_path,
                "annotations" : [
                    {
                        "class_name": class_name,
                        "bbox": box,
                        "segmentation": mask_rle,
                        "score": score,
                    }
                    for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
                ],
                "box_format": "xyxy",
                "img_width": w,
                "img_height": h,
            }
            return jsonify(results)

        with open(output_image_path, "rb") as image_file:
            processed_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        if os.path.exists(output_image_path):
            os.remove(output_image_path)

        return jsonify({
            "labels": labels,
            "masks": masks.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)