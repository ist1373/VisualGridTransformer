import detectron2
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from pydantic import BaseModel, Field
import numpy as np

INSTANCE_PREDICTED_CLASSES_LABEL = "pred_classes"
INSTANCE_SCORES_LABEL = "scores"
INSTANCE_PREDICTED_BOXES_LABEL = "pred_boxes"
INSTANCE_PREDICTED_MASKS_LABEL = "pred_masks"
INSTANCE_INSTANCES_LABEL = "instances"
INSTANCE_DLU_EXTRACTED_LABEL = "dlu_extracted"
INSTANCE_IMAGE_SIZE = "image_size"

class InstancesModel(BaseModel):
        image_size: list[float]
        pred_boxes: list[list]
        scores:list[float]
        pred_classes:list[int]

class InstancesUtils:
    def __init__(self):
        pass

    @staticmethod
    def serialize_instances(instances:Instances, scale = 1) -> dict:
        """
        Convert a Detectron2 Instances object to a serializable dictionary.
        """
        serialized = {
            "image_size": (instances.image_size[0]*scale,instances.image_size[1]*scale),
            "pred_boxes": [[item*scale for item in box.tolist()] for box in instances.pred_boxes] if instances.has("pred_boxes") else [],
            "scores": instances.scores.tolist() if instances.has("scores") else [],
            "pred_classes": instances.pred_classes.tolist() if instances.has("pred_classes") else [],
        }
        return serialized
    

    @staticmethod
    def convert_to_instances(instances_object:InstancesModel):
        """
        Return the annotated object for the image which can later be used by the ocr or
        other things to extract text or other information
        :param output: detectron 2 instaces object
        :param cutoff_threshold: cutoff_threshold, apply a non-maximum suppression threshold on the results.
        By default, it is 0.0
        """
        if instances_object is None:
            # check if the input image is none
            return []
        extracted_layout_objects = detectron2.structures.Instances(
            image_size=(instances_object.image_size[0], instances_object.image_size[1]))
        # apply filtered_instances to the output of the model
        extracted_layout_objects.set(INSTANCE_PREDICTED_CLASSES_LABEL,
                                     np.array(instances_object.pred_classes))
        extracted_layout_objects.set(INSTANCE_SCORES_LABEL,
                                     np.array(instances_object.scores))
        extracted_layout_objects.set(INSTANCE_PREDICTED_BOXES_LABEL,
                                     Boxes(tensor = [box for box in instances_object.pred_boxes]))

        return extracted_layout_objects
    
