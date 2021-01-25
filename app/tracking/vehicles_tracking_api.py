import cv2
import numpy as np 
import time
import logging
import traceback
import os
import io
import requests
import imutils
import itertools
import random
import json
from time import gmtime, strftime
from datetime import datetime

from flask import Flask, render_template, Response, request, jsonify, Blueprint

from utils.parser import get_config
from utils.utils import load_class_names, get_image, get_image_tracking, get_crop_track1
from utils.utils import get_class_vehicle

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

from app.models import db, Camera, Moi, Vehicles, Frames, Type, Color

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
VEHICLE_IMAGE = cfg.SERVICE.VEHICLE_IMAGE

# set up port 
HOST = cfg.SERVICE.SERVICE_IP
PORT = cfg.SERVICE.TRACKING_PORT

# create labels
CLASSES = load_class_names(cfg.DETECTOR.VEHICLE_CLASS)

# setup deepsort
ENCODER = gdet.create_box_encoder(cfg.DEEPSORT.MODEL, batch_size=4)
METRIC = nn_matching.NearestNeighborDistanceMetric("cosine", cfg.DEEPSORT.MAX_COSINE_DISTANCE, cfg.DEEPSORT.NN_BUDGET)
TRACKER = Tracker(METRIC)

LIST_VEHICLE_OUT = []
VIS_CURRENT_FRAME = None
LOGO = 'app/static/figure/logo.png'
LIST_VEHICLE_OUT_PATH = [LOGO] * 6

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

if not os.path.exists(VEHICLE_IMAGE):
    os.mkdir(VEHICLE_IMAGE)

logging.basicConfig(filename=os.path.join(LOG_PATH, str(time.time())+".log"), filemode="w", level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

# app = Flask(__name__)
tracker = Blueprint('tracking', __name__) 

@tracker.route('/track_video', methods=['GET'])
def predict_video():
    # cam_name = 'cam6'
    cam_name = request.args.get('camname')
    path_store_cam = os.path.join(STORE_FRAME, cam_name)
    if not os.path.exists(path_store_cam):
        os.mkdir(path_store_cam)
    
    # create subdir for each class in VEHICLE_IMAGE
    if not os.path.exists(os.path.join(VEHICLE_IMAGE, cam_name)):
        os.mkdir(os.path.join(VEHICLE_IMAGE, cam_name))
        for class_name in CLASSES:
            os.mkdir(os.path.join(VEHICLE_IMAGE, cam_name, class_name))

    cap = cv2.VideoCapture("images/cam1.mp4")
    cnt_frame = 0
    while True:
        ret, frame = cap.read()
        _frame = frame.copy()

        image_name = "frame_" + str(cnt_frame) +  ".jpg"
        print("frame number: ", image_name)
        frame_path = os.path.join(path_store_cam, image_name)
        VIS_CURRENT_FRAME = frame_path
        # store frame
        cv2.imwrite(frame_path, frame)

        image_detect_path = os.path.join(BACKUP, "video_frame.jpg")

        # draw cam's moi and roi 
        moi = cfg.CAM1.MOI
        roi_split_region = cfg.CAM1.ROI_SPLIT_REGION
        frame = draw_ROI(frame, moi, roi_split_region)
        cv2.imwrite(image_detect_path, frame)

        # detection 
        detect_response = requests.post(DETECT_URL, files={"file": ("filename", open(image_detect_path, "rb"), "image/jpeg")}).json()
        vehicle_boxes = detect_response['vehicle_boxes']
        vehicle_scores = detect_response['vehicle_scores']
        vehicle_classes = detect_response['vehicle_classes']

        image, detections = run_detection(frame, vehicle_boxes, vehicle_scores, vehicle_classes, ENCODER, cfg, roi_split_region)
        # tracking
        image, list_vehicle_info, tracker = draw_tracking(image, TRACKER, detections, roi_split_region, cnt_frame, cam_name)

        LIST_VEHICLE_OUT = tracker.que_vehicle_out
        print("##################")
        for i in range(len(LIST_VEHICLE_OUT)):
            LIST_VEHICLE_OUT_PATH[i] = LIST_VEHICLE_OUT[i].path_image
            crop_img_vehicle = cv2.imread(LIST_VEHICLE_OUT_PATH[i])
            crop_img_vehicle = imutils.resize(crop_img_vehicle, width=50)
            cv2.imwrite("vehicle/crop{}/vehicle.jpg".format(i+1), crop_img_vehicle)
        
        print("len list vehicle out path: ", LIST_VEHICLE_OUT_PATH)
        print("##################")
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
                if cam.sum_vehicle <= tracker.counted_track:
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
                        if moi.sum_vehicle <= tracker.list_counted_moi[i]:
                            moi.sum_vehicle = tracker.list_counted_moi[i]
                            db.session.commit()
        
        # add vehicle db
        if cam != None:
            for track in tracker.tracks:
                bbox = track.bbox
                cam_id = cam.id
                vehicle_id = track.track_id
                vehicle = Vehicles.query.filter_by(number_track=vehicle_id).first()
                if vehicle == None:
                    if track.point_in != None:
                        point_in = "{},{}".format(track.point_in[0], track.point_in[1])
                    else:
                        point_in = "0,0"
                    if track.point_out != None:
                        point_out = "{},{}".format(track.point_out[0], track.point_out[1])
                    else:
                        point_out = "0,0"
                    
                    if track.class_name != None:
                        record = Vehicles(vehicle_name=track.class_name, vehicle_score=track.score, vehicle_path=track.path_image, number_track=track.track_id, \
                                        point_in=point_in, point_out=point_out, frame_in=track.frame_in, frame_out=track.frame_out, cam_id=cam_id)
                        db.session.add(record)
                        db.session.commit()

                        # add frames db
                        # bbox = track.bbox
                        bbox = "{},{},{},{}".format(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                        record = Frames(frame_number=cnt_frame, frame_path=track.frame_path, bbox=bbox, vehicle_id=vehicle_id)
                        db.session.add(record)
                        db.session.commit()
 
                else:
                    if track.point_out != None:
                        point_out = "{},{}".format(track.point_out[0], track.point_out[1])
                    else:
                        point_out = "0,0"

                    if track.class_name != None:
                        vehicle.vehicle_name = track.class_name
                        vehicle.vehicle_score = track.score
                        vehicle.vehicle_path = track.path_image
                        vehicle.point_out = point_out
                        vehicle.frame_out = track.frame_out        
                        db.session.commit()
                
                        # add frames db
                        # bbox = track.bbox
                        # print("frame path: ", track.frame_path)
                        bbox = "{},{},{},{}".format(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                        record = Frames(frame_number=cnt_frame, frame_path=track.frame_path, bbox=bbox, vehicle_id=vehicle_id)
                        db.session.add(record)
                        db.session.commit()
                
                if track.flag_attribute == True:
                    # attribute recognition
                    car_response = requests.post(CAR_RECOG_URL, files={"file": ("filename", open(track.path_image, "rb"), "image/jpeg")}).json()
                    attribute = car_response['vehicle_name']
                    car_type = Type.query.filter_by(vehicle_id=vehicle_id).first()
                    if car_type == None:
                        # add type table
                        record = Type(vehicle_type=attribute, vehicle_id=vehicle_id)
                        db.session.add(record)
                        db.session.commit()
                    else:
                        car_type.vehicle_type = attribute
                        db.session.commit()

                    # color recognition
                    color_response = requests.post(COLOR_URL, files={"file": ("filename", open(track.path_image, "rb"), "image/jpeg")}).json()
                    color  = color_response['color']
                    car_color = Color.query.filter_by(vehicle_id=vehicle_id).first()
                    if car_color == None:
                        # add color table
                        record = Color(vehicle_color=color, vehicle_id=vehicle_id)
                        db.session.add(record)
                        db.session.commit()
                    else:
                        car_color.vehicle_color = color
                        db.session.commit()

                    print("add attribute")


        # for vehicle_info in list_vehicle_info:
        #     try:
        #         bbox = vehicle_info['bbox']
        #         str_bbox = "{} {} {} {}".format(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        #         class_name = vehicle_info['class_name']

        #         # vehicle image
        #         vehicle_image = _frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        #         vehicle_image_path = os.path.join(TRACKING_DIR, "vehicle.jpg")
        #         cv2.imwrite(vehicle_image_path, vehicle_image)

        #         # attribute recognition
        #         car_response = requests.post(CAR_RECOG_URL, files={"file": ("filename", open(vehicle_image_path, "rb"), "image/jpeg")}).json()
        #         attribute = car_response['vehicle_name']

        #         # color recognition
        #         color_response = requests.post(COLOR_URL, files={"file": ("filename", open(vehicle_image_path, "rb"), "image/jpeg")}).json()
        #         color  = color_response['color']

                # add type table
                # if cam != None:
                #     record = Vehicle(path=frame_path, name=class_name, box=str_bbox, 
                #                     attribute=attribute, color=color)
                #     db.session.add(record)
                #     db.session.commit()

            # except Exception as e:
            #     logger.error(str(e))
            #     logger.error(str(traceback.print_exc()))
            #     pass
                # result = {'code': '1001', 'status': RCODE.code_1001}
        
        cnt_frame += 1

    return jsonify(result='done')

@tracker.route('/stream1')
def stream_image():
    try:
        image_path = os.path.join(TRACKING_DIR, 'video_frame.jpg')
    except Exception as e:
        print(str(e))
        print(str(traceback.print_exc()))
        result = {'code': '609', 'status': RCODE.code_609}

    return Response(get_image_tracking(image_path),mimetype='multipart/x-mixed-replace; boundary=frame')


@tracker.route('/vehicle/<number_vehicle>')
def stream_vehicle1(number_vehicle):
    try:
        image_path = "vehicle/crop{}/vehicle.jpg".format(number_vehicle)
        
        print("IMG_PATH vehicle1: ", image_path)

    except Exception as e:
        print(str(e))
        print(str(traceback.print_exc()))
        result = {'code': '609', 'status': RCODE.code_609}

    return Response(get_crop_track1(image_path),mimetype='multipart/x-mixed-replace; boundary=frame')

# @tracker.route('/class_vehicle')
# def index():
#     return Response(get_class_vehicle(), content_type='text/event-stream')

    # return redirect(url_for('static', filename='index.html'))
    # return render_template('visual.html')

@tracker.route('/class_vehicle')
def index():
    if request.headers.get('accept') == 'text/event-stream':
        def events():
            for i, c in enumerate(itertools.cycle('\|/-')):
                yield "data: %s %d\n\n" % ('class: ', i)
                time.sleep(.1)  # an artificial delay
        return Response(events(), content_type='text/event-stream')
    # return redirect(url_for('static', filename='index.html'))
    return render_template('visual.html')

@tracker.route('/demo/<number>')
def index2(number):
    if request.headers.get('accept') == 'text/event-stream':
        def events():
            for i, c in enumerate(itertools.cycle('\|/-')):
                yield "data: %s %d %d\n\n" % (c, i, int(number))
                time.sleep(.1)  # an artificial delay
        return Response(events(), content_type='text/event-stream')
    # return redirect(url_for('static', filename='index.html'))
    return render_template('index.html')


# if __name__ == "__main__":
#     app.run(debug=False, host=HOST, port=PORT)