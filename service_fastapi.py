import cv2
import numpy as np 
import time
import logging
import traceback
import os

import tensorflow as tf
import keras 
from keras.models import load_model

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from utils.parser import get_config
from src.predict_color import predict

# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')
cfg.merge_from_file('configs/rcode.yaml')

MODEL = load_model(cfg.SERVICE.COLOR_MODEL)
COLOR_LABELS = cfg.SERVICE.COLOR_LABELS
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
    color: str 

# Define the main route
@app.get('/')
def root_route():
    return {'error': 'Use Get /predict instead of the root route!'}

@app.post('/predict', response_model=Prediction)
async def predict_color(file: UploadFile = File(...)):
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        contents = await file.read()
        image = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # predict color
        my_color = predict(image, MODEL, COLOR_LABELS)

        result = {"code": "1000", "color": my_color}
        return result

    except Exception as e:
        logger.error(str(e))
        logger.error(str(traceback.print_exc()))
        
        raise HTTPException(status_code=500, detail=str(e))