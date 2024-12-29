from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from subprocess import run, CalledProcessError
import os
from src.object_detection.pdf2img import convert_pdf_to_image
from src.object_detection.create_grid_input import create_grid
from src.object_detection.inference import process_bounding_boxes
from src.object_detection.draw_instances import draw_instances, draw_ordered_instances
from src.object_detection.instances_utils import InstancesUtils
from src.object_detection.scene import Scene
from fastapi.responses import ORJSONResponse
import logging
from fastapi import FastAPI, File, UploadFile, Form
import json
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


app = FastAPI()

class Pdf2ImgArgs(BaseModel):
    pdf: str
    output: str
    format: str = Field(default="png", description="image format")
    low_dpi: int = Field(default=72, description="low_dpi")
    high_dpi: int = Field(default=216, description="high_dpi")

class CreateGridArgs(BaseModel):
    pdf: str = Field(default="default.pdf", description="Path to the input PDF file")
    output: str = Field(default="output", description="Path to the output directory")
    tokenizer: str = Field(default="/Users/imansaberi/Documents/research_projects/content_extraction/VGT/model/tokenizer/layoutlm-base-uncased/", description="Address of the tokenizer")
    model: str = Field(default="publaynet", description="model name")



@app.post("/convert-pdf-to-img/")
def convert_pdf_to_img(args: Pdf2ImgArgs):
    """
    Endpoint to convert a PDF file to images using the pdf2img.py script.
    
    Args:
        args: Arguments required for the pdf2img.py script:
            - pdf (str): Path to the input PDF file.
            - output (str): Directory to save the output images.
            - format (str): Image format for output files (e.g., png, jpg).
            - dpi (int): DPI (dots per inch) for image resolution.

    Returns:
        JSON with status and output path or an error message.
    """

    if not os.path.exists(args.pdf):
        raise HTTPException(status_code=400, detail="PDF file does not exist.")

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    try:
        convert_pdf_to_image(args.output,args.pdf,args.format,args.low_dpi)
        convert_pdf_to_image(args.output,args.pdf,args.format,args.high_dpi,prefix="high")
        return {"status": "success", f"output": "images has been created in {args.output}"}
    except CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error: {e.stderr}")


@app.post("/create-grid-info/")
def create_grid_info(args: CreateGridArgs):

    # Validate and sanitize paths
    args.pdf = os.path.abspath(args.pdf)
    args.output = os.path.abspath(args.output)
    args.tokenizer = os.path.abspath(args.tokenizer)

    logging.info("Received request with args: %s", args.dict())

    if not os.path.exists(args.pdf):
        logging.error("PDF file does not exist: %s", args.pdf)
        raise HTTPException(status_code=400, detail="PDF file does not exist.")

    # Ensure output directory exists
    if not os.path.exists(args.output):
        try:
            os.makedirs(args.output, exist_ok=True)
            logging.info("Created output directory: %s", args.output)
        except OSError as e:
            logging.error("Failed to create output directory: %s", args.output)
            raise HTTPException(status_code=500, detail="Failed to create output directory.")
    try:
        create_grid(args.pdf, args.output, args.tokenizer, args.model)
        logging.info("Grid creation successful, saved to: %s", args.output)
        return {
            "status": "success",
            "output": f"Grids have been successfully saved in {args.output}"
        }
    except CalledProcessError as e:
        logging.error("Error during grid creation: %s", e.stderr)
        raise HTTPException(status_code=500, detail=f"Error occurred while processing: {e.stderr}")
    except Exception as e:
        logging.error("Unexpected error: %s", str(e))
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")



class ExtractBoundingBoxesArgs(BaseModel):
    image_root: str = Field(default="./outputs/", description="Root directory for the images")
    grid_root: str = Field(default="./outputs/", description="Root directory for the grids")
    image_name: str = Field(default="page_1", description="Name of the image to process")
    dataset: str = Field(default="publaynet", description="Name of the dataset")
    output_root: str = Field(default="./outputs/", description="Root directory for the output")
    config: str = Field(default="./src/object_detection/Configs/cascade/publaynet_VGT_cascade_PTM.yaml", description="Path to the configuration file")
    high_res_image_path: str = Field(default=None, description="Root directory for the images")


@app.get("/extract-bounding-boxes/")
def extract_bounding_boxes(args: ExtractBoundingBoxesArgs):

    logging.info("Received request with args: %s", args.dict())

    # Ensure output directory exists
    if not os.path.exists(args.output_root):
        try:
            os.makedirs(args.output_root, exist_ok=True)
            logging.info("Created output directory: %s", args.output_root)
        except OSError as e:
            logging.error("Failed to create output directory: %s", args.output_root)
            raise HTTPException(status_code=500, detail="Failed to create output directory.")
    try:
        output = process_bounding_boxes(args.image_root,args.grid_root,args.image_name,args.output_root,args.dataset,args.config,args.high_res_image_path)
        logging.info("extract bounding boxes successful, saved to: %s", args.output_root)
        print(output)
        return ORJSONResponse({
            "status": "success",
            "output": output
        })
    except CalledProcessError as e:
        logging.error("Error during grid creation: %s", e.stderr)
        raise HTTPException(status_code=500, detail=f"Error occurred while processing: {e.stderr}")
    except Exception as e:
        logging.error("Unexpected error: %s", str(e))
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


class InstancesModel(BaseModel):
    image_size: list[float]
    pred_boxes: list[list]
    scores:list[float]
    pred_classes:list[int]

class SceneModel(BaseModel):
    components: list[list[float]]
    x1: float
    y1: float
    x2: float
    y2: float
    zoom_factor:float

    
class DrawInstancesArgs(BaseModel):
    image_path: str = Field(default="./outputs/", description="Root directory for the images")
    output_file_name: str = Field(default="./outputs/", description="Root directory for the output")
    config: str = Field(default="./src/object_detection/Configs/cascade/publaynet_VGT_cascade_PTM.yaml")
    scenes:list[SceneModel]= Field(default=None, description="scenes")
    json_instances: InstancesModel

@app.post("/draw-instances/")
async def draw_instances_image(args: DrawInstancesArgs):
    try:
        # Save the uploaded file to a temporary location
        
        instances = InstancesUtils.convert_to_instances(args.json_instances)
        # draw_instances(args.image_path,args.output_file_name,instances,args.config)
        scenes = []
        for scene in args.scenes:
            scenes.append(Scene(scene.x1,scene.y1,scene.x2,scene.y2,scene.components,scene.zoom_factor))

        draw_ordered_instances(args.output_file_name,instances,scenes)
        # Return the output file as a response
        return ORJSONResponse({
            "status": "success"
        })
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=e)


if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)