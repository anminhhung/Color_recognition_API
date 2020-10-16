import cv2
import numpy as np 
import time
import logging
import traceback
import os
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from utils.parser import get_config
from utils.utils import load_class_names
from src.vehicle_detection import predict

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg as config_detectron

# create visual dir
if not os.path.exists('visual'):
    os.mkdir('visual')
    
# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')
cfg.merge_from_file('configs/rcode.yaml')

path_weigth = cfg.SERVICE.DETECT_WEIGHT
path_config = cfg.SERVICE.DETECT_CONFIG
confidences_threshold = cfg.SERVICE.THRESHOLD
num_of_class = cfg.SERVICE.NUMBER_CLASS

# create labels
CLASSES = load_class_names(cfg.SERVICE.VEHICLE_CLASS)

# set up detectron
detectron = config_detectron()
detectron.MODEL.DEVICE= cfg.SERVICE.DEVICE
detectron.merge_from_file(path_config)
detectron.MODEL.WEIGHTS = path_weigth

detectron.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidences_threshold
detectron.MODEL.ROI_HEADS.NUM_CLASSES = num_of_class

PREDICTOR = DefaultPredictor(detectron)

# create log_file, rcode
HOST = cfg.SERVICE.SERVICE_IP
PORT = cfg.SERVICE.DETECTION_PORT
LOG_PATH = cfg.SERVICE.LOG_PATH
RCODE = cfg.RCODE

# create logging
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
logging.basicConfig(filename=os.path.join(LOG_PATH, str(time.time())+".log"), filemode="w", level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

app = FastAPI()

# Define the Response
class Prediction(BaseModel):
    code: str 
    visual_path: str
    # vehicle_paths: list
    vehicle_boxes: list
    vehicle_scores: list
    vehicle_classes: list

@app.post('/predict', response_model=Prediction)
async def predict_car(file: UploadFile = File(...)):
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')
    try:
        contents = await file.read()
        image = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # detect
        visual_path, list_boxes, list_scores, list_classes = predict(image, PREDICTOR, CLASSES)

        result = {"code": "1000", "visual_path": visual_path, "vehicle_boxes": list_boxes, "vehicle_scores": list_scores, "vehicle_classes": list_classes}
        with open("demo.txt", "a+") as f:
            f.write("{}\n".format(result))
        return result

    except Exception as e:
        logger.error(str(e))
        logger.error(str(traceback.print_exc()))
        
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("vehicles_detection_api:app", host=HOST, port=PORT)