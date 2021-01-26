from flask import Flask, render_template, request, jsonify

import cv2
import numpy as np 
import argparse
import os
import glob
import random
import darknet
import time
import numpy as np
import darknet
from yolov4 import predict_yolov4

import cv2
import numpy as np 
import time
import logging
import traceback
import os
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

app = FastAPI()

network, class_names, class_colors = darknet.load_network(
        "yolov4_416.cfg",
        "traffic.data",
        "yolov4_416_last.weights",
        batch_size=8
)

# Define the Response
class Prediction(BaseModel):
    code: str 
    # vehicle_paths: list
    vehicle_boxes: list
    vehicle_scores: list
    vehicle_classes: list

@app.post('/detect_traffic', response_model=Prediction)
async def predict_car(file: UploadFile = File(...)):
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')
    try:
        contents = await file.read()
        image = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # detect detectron
        # list_boxes, list_scores, list_classes = predict(image, PREDICTOR, CLASSES)

        # detect yolov4
        list_bboxes, list_scores, list_labels = predict_yolov4(image, network, class_names, class_colors)

        result = {"code": "1000", "vehicle_boxes": list_bboxes, "vehicle_scores": list_scores, "vehicle_classes": list_labels}

        return result

    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("vehicles_detection_api:app", host='0.0.0.0', port=7003)

