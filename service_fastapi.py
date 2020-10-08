import cv2
import numpy as np 
import time
import logging
import traceback
import os
import requests
import random
from time import gmtime, strftime
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from utils.parser import get_config
from utils.utils import load_class_names, predict_all_feature

# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')
cfg.merge_from_file('configs/rcode.yaml')

# create backup dir
if not os.path.exists('backup'):
    os.mkdir('backup')

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
    vehicle_path: str
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

        # gen name
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        number = str(random.randint(0, 10000))
        img_name = time + '_' + number + '.jpg'
        img_path = os.path.join('backup', img_name)
        cv2.imwrite(img_path, image)

        # detect
        detect_response = requests.post('http://0.0.0.0:5003/predict', files={"file": ("filename", open(img_path, "rb"), "image/jpeg")}).json()
        vehicle_paths = detect_response['vehicle_paths']
        vehicle_scores = detect_response['vehicle_scores']
        vehicle_classes = detect_response['vehicle_classes']

        # color recognition
        color_response = requests.post('http://0.0.0.0:5001/predict', files={"file": ("filename", open(vehicle_paths[0], "rb"), "image/jpeg")}).json()
        vehicle_color  = color_response['color']

        # car recognition
        car_response = requests.post('http://0.0.0.0:5002/predict', files={"file": ("filename", open(vehicle_paths[0], "rb"), "image/jpeg")}).json()
        vehicle_name = car_response['vehicle_name']

        result = {
            "code": "1000", 
            "vehicle_path": vehicle_paths[0], 
            "vehicle_score": vehicle_scores[0], 
            "vehicle_class": vehicle_classes[0],
            "vehicle_color": vehicle_color,
            "vehicle_name": vehicle_name
        }

        return result

    except Exception as e:
        logger.error(str(e))
        logger.error(str(traceback.print_exc()))
        
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("service_fastapi:app", host="0.0.0.0", port=5050)