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
from datetime import datetime

from flask import Flask, render_template, Response, request, jsonify, Blueprint

from utils.parser import get_config
from utils.utils import load_class_names, get_image, get_image_tracking

from src import detect
from src import run_detection, draw_tracking
from utils.parser import get_config
from utils.utils import draw_ROI
from app.tracking.counting import check_in_polygon

from libs import preprocessing
from libs import nn_matching
from libs import Detection
from libs import Tracker
from utils import generate_detections as gdet
from libs import Detection as ddet
from collections import deque

from src import detect

from app.models import db, Camera, Moi, Traffic, Vehicles, Frames, Type, Color

# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')
cfg.merge_from_file('configs/rcode.yaml')
cfg.merge_from_file('configs/detect.yaml')
cfg.merge_from_file('configs/deepsort.yaml')
cfg.merge_from_file('configs/cam.yaml')

# create log_file, rcode
COLOR_URL = cfg.SERVICE.COLOR_URL
DETECT_URL = cfg.SERVICE.DETECT_URL
CAR_RECOG_URL = cfg.SERVICE.CAR_RECOG_URL
LOG_PATH = cfg.SERVICE.LOG_PATH
RCODE = cfg.RCODE
BACKUP = cfg.SERVICE.BACKUP_DIR
STORE_FRAME = cfg.SERVICE.STORE_FRAME

# set up port 
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

if not os.path.exists(STORE_FRAME):
    os.mkdir(STORE_FRAME)

logging.basicConfig(filename=os.path.join(LOG_PATH, str(time.time())+".log"), filemode="w", level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

# app = Flask(__name__)
tracker = Blueprint('tracking', __name__) 

@tracker.route('/predict', methods=['GET'])
def predict_video():
    cam_name = 'cam6'
    cap = cv2.VideoCapture("images/cam6.mp4")
    # add cam_name to camera table
    # try:
    #     record = Camera(cam_name=cam_name)
    #     db.session.add(record)
    #     db.session.commit()
    # except Exception as e:
    #     print(e)
    #     pass 
    
    # # query cam
    # cam = Camera.query.get(cam_name)
    # print("Cam: ", cam)

    while True:
        ret, frame = cap.read()
        _frame = frame.copy()
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        image_name = "image_" + dt_string + "_" + str(random.randint(0, 1000)) +  ".jpg"
        print(image_name)
        frame_path = os.path.join(STORE_FRAME, image_name)
        # store frame
        cv2.imwrite(frame_path, frame)

        image_detect_path = os.path.join(BACKUP, "video_frame.jpg")

        # draw cam's moi and roi 
        moi = cfg.CAM6.MOI
        roi_split_region = cfg.CAM6.ROI_SPLIT_REGION
        frame = draw_ROI(frame, moi, roi_split_region)
        cv2.imwrite(image_detect_path, frame)

        # detection 
        # list_boxes, list_scores, list_classes = detect(frame, net, output_layers, classes)
        detect_response = requests.post(DETECT_URL, files={"file": ("filename", open(image_detect_path, "rb"), "image/jpeg")}).json()
        vehicle_boxes = detect_response['vehicle_boxes']
        vehicle_scores = detect_response['vehicle_scores']
        vehicle_classes = detect_response['vehicle_classes']

        image, detections = run_detection(frame, vehicle_boxes, vehicle_scores, vehicle_classes, ENCODER, cfg, roi_split_region)
        # tracking
        image, list_vehicle_info, tracker = draw_tracking(image, TRACKER, detections, roi_split_region)

        # query cam
        print("##################")
        cam = Camera.query.filter_by(cam_name=cam_name).first()
        print("Cam: ", cam)
        print("##################")

        # add InfoCam db here
        if tracker.counted_track != 0:
            if cam == None:
                record = Camera(cam_name=cam_name, sum_vehicle=tracker.counted_track, sum_xe_may=1, sum_ba_gac=1, sum_taxi=1, sum_car=1,\
                                sum_ban_tai=1, sum_cuu_thuong=1, sum_xe_khach=1, sum_bus=1, sum_tai=1, sum_container=1)
                db.session.add(record)
                db.session.commit()
            else:
                if cam.sum_vehicle != tracker.counted_track:
                    cam.sum_vehicle = tracker.counted_track
                    db.session.commit()
        
        # add moi db:
        if cam != None:
            for i in range(len(tracker.list_counted_moi)):
                cam_id = cam.id
                moi = Moi.query.filter_by(cam_id=cam_id).first()
                if moi == None:
                    record = Moi(moi_name=str(i+1), sum_vehicle=tracker.list_counted_moi[i], sum_xe_may=1, sum_ba_gac=1, sum_taxi=1, sum_car=1,\
                                sum_ban_tai=1, sum_cuu_thuong=1, sum_xe_khach=1, sum_bus=1, sum_tai=1, sum_container=1, cam_id=cam_id)
                    db.session.add(record)
                    db.session.commit()
                else:
                    moi = Moi.query.filter_by(cam_id=cam_id, moi_name=str(i+1)).first()
                    if moi == None:
                        record = Moi(moi_name=str(i+1), sum_vehicle=tracker.list_counted_moi[i], sum_xe_may=1, sum_ba_gac=1, sum_taxi=1, sum_car=1,\
                                sum_ban_tai=1, sum_cuu_thuong=1, sum_xe_khach=1, sum_bus=1, sum_tai=1, sum_container=1, cam_id=cam_id)
                        db.session.add(record)
                        db.session.commit()
                    else:
                        if moi.sum_vehicle != tracker.list_counted_moi[i]:
                            moi.sum_vehicle = tracker.list_counted_moi[i]
                            db.session.commit()

        for vehicle_info in list_vehicle_info:
            try:
                bbox = vehicle_info['bbox']
                str_bbox = "{} {} {} {}".format(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                class_name = vehicle_info['class_name']

                # vehicle image
                vehicle_image = _frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                vehicle_image_path = os.path.join(TRACKING_DIR, "vehicle.jpg")
                cv2.imwrite(vehicle_image_path, vehicle_image)

                # attribute recognition
                car_response = requests.post(CAR_RECOG_URL, files={"file": ("filename", open(vehicle_image_path, "rb"), "image/jpeg")}).json()
                attribute = car_response['vehicle_name']

                # color recognition
                color_response = requests.post(COLOR_URL, files={"file": ("filename", open(vehicle_image_path, "rb"), "image/jpeg")}).json()
                color  = color_response['color']
                
                # add record to db
                # record = Vehicle(path=frame_path, name=class_name, box=str_bbox, 
                #                 attribute=attribute, color=color)
                # db.session.add(record)
                # db.session.commit()

            except Exception as e:
                logger.error(str(e))
                logger.error(str(traceback.print_exc()))
                # result = {'code': '1001', 'status': RCODE.code_1001}

    return jsonify(result='done')

@tracker.route('/stream')
def stream_image():
    try:
        image_path = os.path.join(TRACKING_DIR, 'video_frame.jpg')
    except Exception as e:
        print(str(e))
        print(str(traceback.print_exc()))
        result = {'code': '609', 'status': RCODE.code_609}

    return Response(get_image_tracking(image_path),mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=False, host=HOST, port=PORT)