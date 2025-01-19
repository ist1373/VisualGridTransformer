import argparse
import os
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances

from src.object_detection.ditod.VGTTrainer import DefaultPredictor
from src.object_detection.ditod import add_vit_config
import pickle

class VGT:
    def __init__(self):
        self.cfg = get_cfg()
        add_vit_config(self.cfg)
        config_file = "./src/object_detection/Configs/cascade/publaynet_VGT_cascade_PTM.yaml"
        self.cfg.merge_from_file(config_file)
        
        # Step 3: set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.WEIGHTS = "/Users/imansaberi/Documents/research_projects/content_extraction/VGT/model/publaynet_VGT_model.pth"

        # Step 4: define model
        self.predictor = DefaultPredictor(self.cfg)

    def serialize_instances(self,instances: Instances, scale = 1) -> dict:
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

    def process_bounding_boxes(self,image_root,grid_root,image_name,output_root,dataset, high_res_img_path = None,opts=[]):

        image_path = os.path.join(image_root, f"{image_name}.png")
        grid_extensions = {
            'publaynet': ".pdf.pkl",
            'docbank': ".pkl",
            'D4LA': ".pkl",
            'doclaynet': ".pdf.pkl"
        }
        grid_path = os.path.join(grid_root, f"{image_name}{grid_extensions[dataset]}")
        output_file_name = os.path.join(output_root, f"{image_name}_output.png")


        
        # Step 5: run inference
        img = cv2.imread(image_path)
        
        md = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        if dataset == 'publaynet':
            md.set(thing_classes=["text","title","list","table","figure"])
        elif dataset == 'docbank':
            md.set(thing_classes=["abstract","author","caption","date","equation", "figure", "footer", "list", "paragraph", "reference", "section", "table", "title"])
        elif dataset == 'D4LA':
            md.set(thing_classes=["DocTitle","ParaTitle","ParaText","ListText","RegionTitle", "Date", "LetterHead", "LetterDear", "LetterSign", "Question", "OtherText", "RegionKV", "Regionlist", "Abstract", "Author", "TableName", "Table", "Figure", "FigureName", "Equation", "Reference", "Footnote", "PageHeader", "PageFooter", "Number", "Catalog", "PageNumber"])
        elif dataset == 'doclaynet':
            md.set(thing_classes=["Caption","Footnote","Formula","List-item","Page-footer", "Page-header", "Picture", "Section-header", "Table", "Text", "Title"])

        output = self.predictor(img, grid_path)["instances"]
        
        # # import ipdb;ipdb.set_trace()
        # v = Visualizer(img[:, :, ::-1],
        #                 md,
        #                 scale=1.0,
        #                 instance_mode=ColorMode.SEGMENTATION)
        # result = v.draw_instance_predictions(output.to("cpu"))
        # result_image = result.get_image()[:, :, ::-1]

        # cv2.imwrite(output_file_name, result_image)


        # with open("./temp.pkl", "wb") as file:  # "wb" mode is for writing in binary
        #     pickle.dump(output, file)
        serialize_outputs = self.serialize_instances(output)

        
        if high_res_img_path:
            high_img = cv2.imread(high_res_img_path)
            scale = high_img.shape[0] / img.shape[0]

            serialize_outputs = self.serialize_instances(output,scale=scale)

            output.pred_boxes = [[item*scale for item in box.tolist()] for box in output.pred_boxes]

            v = Visualizer(high_img[:, :, ::-1],
                    md,
                    scale=1.0,
                    instance_mode=ColorMode.SEGMENTATION)
            result = v.draw_instance_predictions(output.to("cpu"))
            result_image = result.get_image()[:, :, ::-1]

            cv2.imwrite(output_file_name, result_image)

            return serialize_outputs
        return serialize_outputs


def main():
    parser = argparse.ArgumentParser(description="Detectron2 inference script")
    parser.add_argument(
        "--image_root",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--grid_root",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--image_name",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_root",
        help="Name of the output visualization file.",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    process_bounding_boxes(args.image_root,args.grid_root,args.image_name,args.output_root,args.dataset,args.config_file,opts=[])


if __name__ == '__main__':
    main()

