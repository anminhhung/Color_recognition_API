  
import cv2
import numpy as np 
import time
import logging
import traceback
import os
import uvicorn

import tensorflow as tf
import keras 
from keras.models import model_from_json

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from utils.parser import get_config
from utils.utils import load_class_names
from src.predict_car import predict

# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')
cfg.merge_from_file('configs/rcode.yaml')

with open(cfg.SERVICE.CAR_RECOG_JSON, 'r') as json_file:
    model_json = json_file.read()

# Load weights
WEIGHTS = cfg.SERVICE.CAR_RECOG_WEIGHTS
MODEL = model_from_json(model_json)
MODEL.load_weights(WEIGHTS)

HOST = cfg.SERVICE.SERVICE_IP
PORT = cfg.SERVICE.CAR_RECOG_PORT
LOG_PATH = cfg.SERVICE.LOG_PATH
RCODE = cfg.RCODE
# create labels
LABELS = load_class_names(cfg.SERVICE.CAR_RECOG_LABELS)

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
    vehicle_name: str 

@app.post('/predict', response_model=Prediction)
async def predict_car(file: UploadFile = File(...)):
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        contents = await file.read()
        image = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # predict color
        car_name = predict(image, MODEL, LABELS)

        result = {"code": "1000", "vehicle_name": car_name}
        return result

    except Exception as e:
        logger.error(str(e))
        logger.error(str(traceback.print_exc()))
        
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("vehicles_recog_api:app", host=HOST, port=PORT)