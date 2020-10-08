import cv2
import numpy as np 
import time
import logging
import traceback
import os

import tensorflow as tf
import keras 
from keras.models import load_model, model_from_json

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from utils.parser import get_config
from utils.utils import load_class_names, predict_all_feature
from src.vehicle_detection import predict

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg as config_detectron

# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')
cfg.merge_from_file('configs/rcode.yaml')

# set up detectron
path_weigth = cfg.SERVICE.DETECT_WEIGHT
path_config = cfg.SERVICE.DETECT_CONFIG
confidences_threshold = cfg.SERVICE.THRESHOLD
num_of_class = cfg.SERVICE.NUMBER_CLASS

VEHICLE_CLASSES = load_class_names(cfg.SERVICE.VEHICLE_CLASS)

detectron = config_detectron()
detectron.MODEL.DEVICE= cfg.SERVICE.DEVICE
detectron.merge_from_file(path_config)
detectron.MODEL.WEIGHTS = path_weigth

detectron.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidences_threshold
detectron.MODEL.ROI_HEADS.NUM_CLASSES = num_of_class

PREDICTOR = DefaultPredictor(detectron)
############################

# set up color recognition
COLOR_MODEL = load_model(cfg.SERVICE.COLOR_MODEL)
COLOR_LABELS = cfg.SERVICE.COLOR_LABELS
############################

# set up car recognition
with open(cfg.SERVICE.CAR_RECOG_JSON, 'r') as json_file:
    model_json = json_file.read()

CAR_WEIGHTS = cfg.SERVICE.CAR_RECOG_WEIGHTS
CAR_MODEL = model_from_json(model_json)
CAR_MODEL.load_weights(CAR_WEIGHTS)
CAR_LABELS = load_class_names(cfg.SERVICE.CAR_RECOG_LABELS)
############################

# create log_file, rcode
HOST = cfg.SERVICE.SERVICE_IP
PORT = cfg.SERVICE.SERVICE_PORT
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
    vehicle_box: list
    vehicle_score: float
    vehicle_class: str
    vehicle_color: str 
    vehicle_name: str

@app.post('/predict', response_model=Prediction)
async def predict_car(file: UploadFile = File(...)):
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        contents = await file.read()
        image = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # detect
        vehicle_box, vehicle_score, vehicle_class, vehicle_color, vehicle_name = predict_all_feature(
                                                                    image, PREDICTOR, CAR_MODEL, COLOR_MODEL,
                                                                    CAR_LABELS, COLOR_LABELS, VEHICLE_CLASSES
                                                                )

        result = {
            "code": "1000", 
            "vehicle_box": vehicle_box, 
            "vehicle_score": vehicle_score, 
            "vehicle_class": vehicle_class,
            "vehicle_color": vehicle_color,
            "vehicle_name": vehicle_name
        }

        return result

    except Exception as e:
        logger.error(str(e))
        logger.error(str(traceback.print_exc()))
        
        raise HTTPException(status_code=500, detail=str(e))