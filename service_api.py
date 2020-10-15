import cv2
import numpy as np 
import time
import logging
import traceback
import os
import io
import requests
import random
import json
import uvicorn
from time import gmtime, strftime

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from utils.parser import get_config
from utils.utils import load_class_names, get_frame

# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')
cfg.merge_from_file('configs/rcode.yaml')

# create backup dir
if not os.path.exists('backup'):
    os.mkdir('backup')

# create json dir
if not os.path.exists('json_dir'):
    os.mkdir('json_dir')

# create log_file, rcode
COLOR_URL = cfg.SERVICE.COLOR_URL
DETECT_URL = cfg.SERVICE.DETECT_URL
CAR_RECOG_URL = cfg.SERVICE.CAR_RECOG_URL
LOG_PATH = cfg.SERVICE.LOG_PATH
RCODE = cfg.RCODE
BACKUP = cfg.SERVICE.BACKUP_DIR


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
    vehicles: list

@app.post('/predict', response_model=Prediction)
async def predict_car(file: UploadFile = File(...)):
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        contents = await file.read()
        image = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image_path = os.path.join(BACKUP, file.filename)


        # cv2.imwrite(image_path, image)

        # detect
        detect_response = requests.post(DETECT_URL, files={"file": ("filename", open(image_path, "rb"), "image/jpeg")}).json()

        visual_path = detect_response['visual_path']
        vehicle_paths = detect_response['vehicle_paths']
        vehicle_scores = detect_response['vehicle_scores']
        vehicle_classes = detect_response['vehicle_classes']

        list_vehicle = []
        cnt = 0
        for vehicle_path in vehicle_paths:
            # color recognition
            color_response = requests.post(COLOR_URL, files={"file": ("filename", open(vehicle_path, "rb"), "image/jpeg")}).json()
            vehicle_color  = color_response['color']

            # car recognition
            car_response = requests.post(CAR_RECOG_URL, files={"file": ("filename", open(vehicle_path, "rb"), "image/jpeg")}).json()
            vehicle_name = car_response['vehicle_name']

            result = {
                "vehicle_path": vehicle_path, 
                "vehicle_score": vehicle_scores[cnt], 
                "vehicle_class": vehicle_classes[cnt],
                "vehicle_color": vehicle_color,
                "vehicle_name": vehicle_name
                }

            cnt += 1
            list_vehicle.append(result)
        
        result = {"code": "1000", "visual_path": visual_path, "vehicles": list_vehicle}

        return result

    except Exception as e:
        logger.error(str(e))
        logger.error(str(traceback.print_exc()))
        
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("service_api:app", host="0.0.0.0", port=5050)