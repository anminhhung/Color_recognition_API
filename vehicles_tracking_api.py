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
from time import gmtime, strftime

from flask import Flask, render_template, Response, request, jsonify

from utils.parser import get_config
from utils.utils import load_class_names, get_image, get_image_tracking

from src import detect
from src import run_detection, draw_tracking
from utils.parser import get_config

from libs import preprocessing
from libs import nn_matching
from libs import Detection
from libs import Tracker
from utils import generate_detections as gdet
from libs import Detection as ddet
from collections import deque

from src import detect

# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')
cfg.merge_from_file('configs/rcode.yaml')
cfg.merge_from_file('configs/detect.yaml')
cfg.merge_from_file('configs/deepsort.yaml')

# create log_file, rcode
COLOR_URL = cfg.SERVICE.COLOR_URL
DETECT_URL = cfg.SERVICE.DETECT_URL
CAR_RECOG_URL = cfg.SERVICE.CAR_RECOG_URL
LOG_PATH = cfg.SERVICE.LOG_PATH
RCODE = cfg.RCODE
BACKUP = cfg.SERVICE.BACKUP_DIR
HOST = cfg.SERVICE.SERVICE_IP
PORT = cfg.SERVICE.TRACKING_PORT

# setup deepsort
ENCODER = gdet.create_box_encoder(cfg.DEEPSORT.MODEL, batch_size=4)
METRIC = nn_matching.NearestNeighborDistanceMetric("cosine", cfg.DEEPSORT.MAX_COSINE_DISTANCE, cfg.DEEPSORT.NN_BUDGET)
TRACKER = Tracker(METRIC)

# create output_detect dir
TRACKING_DIR = 'output_tracking'
if not os.path.exists(TRACKING_DIR):
    os.mkdir(TRACKING_DIR)

# create backup dir
if not os.path.exists(BACKUP):
    os.mkdir(BACKUP)

if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
logging.basicConfig(filename=os.path.join(LOG_PATH, str(time.time())+".log"), filemode="w", level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict_video():
    cap = cv2.VideoCapture("images/cam_02.mp4")
    while True:
        ret, frame = cap.read()

        image_detect_path = os.path.join(BACKUP, "video_frame.jpg")
        cv2.imwrite(image_detect_path, frame)

        # detection 
        # list_boxes, list_scores, list_classes = detect(frame, net, output_layers, classes)
        detect_response = requests.post(DETECT_URL, files={"file": ("filename", open(image_detect_path, "rb"), "image/jpeg")}).json()
        vehicle_boxes = detect_response['vehicle_boxes']
        vehicle_scores = detect_response['vehicle_scores']
        vehicle_classes = detect_response['vehicle_classes']

        image, detections = run_detection(frame, vehicle_boxes, vehicle_scores, vehicle_classes, ENCODER, cfg)
        # tracking
        image = draw_tracking(image, TRACKER, detections)
    
    return jsonify(result='done')

@app.route('/stream')
def stream_image():
    try:
        image_path = os.path.join(TRACKING_DIR, 'video_frame.jpg')
    except Exception as e:
        print(str(e))
        print(str(traceback.print_exc()))
        result = {'code': '609', 'status': RCODE.code_609}

    return Response(get_image_tracking(image_path),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=False, host=HOST, port=PORT)