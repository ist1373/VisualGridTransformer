import argparse
import os
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances
from detectron2.structures import Instances, Boxes

from src.object_detection.ditod.VGTTrainer import DefaultPredictor
from src.object_detection.ditod import add_vit_config
import pickle
import numpy as np



# def json_to_instances(json_data):
#     """
#     Convert JSON data into a Detectron2 Instances object.

#     Args:
#         json_data (dict): JSON-like dictionary with image_size, pred_boxes, scores, and pred_classes.

#     Returns:
#         detectron2.structures.Instances: Instances object created from the JSON data.
#     """
#     # Extract data from the JSON object
#     image_size = tuple(json_data["output"]["image_size"])
#     pred_boxes = json_data["output"]["pred_boxes"]
#     scores = json_data["output"]["scores"]
#     pred_classes = json_data["output"]["pred_classes"]
    
#     # Convert to torch tensors
#     pred_boxes = Boxes(torch.tensor(pred_boxes, dtype=torch.float32))  # Boxes must be wrapped in Boxes()
#     scores = torch.tensor(scores, dtype=torch.float32)
#     pred_classes = torch.tensor(pred_classes, dtype=torch.int64)
    
#     # Create Instances object
#     instances = Instances(image_size)
#     instances.pred_boxes = pred_boxes
#     instances.scores = scores
#     instances.pred_classes = pred_classes
    
#     return instances

def draw_instances(image_path,output_file_name,instances:Instances,config_file):
    img = cv2.imread(image_path)
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(config_file)
    
    # Step 2: add model weights URL to config
    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    md.set(thing_classes=["text","title","list","table","figure"])

    v = Visualizer(img[:, :, ::-1],
                    md,
                    scale=1.0,
                    instance_mode=ColorMode.SEGMENTATION)
    result = v.draw_instance_predictions(instances)
    result_image = result.get_image()[:, :, ::-1]
    
    # step 6: save
    cv2.imwrite(output_file_name, result_image)

def draw_ordered_instances(output_file_name, instances):

    # Extract image dimensions from the instances object
    height, width = instances.image_size
    
    # Create a white image
    white_image = np.ones((int(height), int(width), 3), dtype=np.uint8) * 255
    
    # Extract bounding boxes
    boxes = instances.pred_boxes.tensor.cpu().numpy()  # Convert to numpy array
    
    # Draw each rectangle
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        # Draw rectangle on the white image
        cv2.rectangle(white_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Add the order as text near the rectangle
        cv2.putText(
            white_image, str(i + 1), (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
        )
    cv2.imwrite(output_file_name, white_image)

