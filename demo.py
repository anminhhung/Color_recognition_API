import cv2
import os
import numpy as np 
import requests

from src import detect
from src import run_detection, draw_tracking
from utils.parser import get_config

from PIL import Image

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

net = cv2.dnn.readNet("models/yolov4.weights", "models/yolov4.cfg")
# Name custom object
classes = ['Loai1', 'Loai2', 'Loai3', 'Loai4', 'Loai5']

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

ENCODER = gdet.create_box_encoder(cfg.DEEPSORT.MODEL, batch_size=4)
METRIC = nn_matching.NearestNeighborDistanceMetric("cosine", cfg.DEEPSORT.MAX_COSINE_DISTANCE, cfg.DEEPSORT.NN_BUDGET)
TRACKER = Tracker(METRIC)

def run_video(video_path, net, output_layers, cfg, classes):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()

        image_detect_path = os.path.join("backup", "video_frame.jpg")
        cv2.imwrite(image_detect_path, frame)

        # detection 
        # list_boxes, list_scores, list_classes = detect(frame, net, output_layers, classes)
        detect_response = requests.post('http://0.0.0.0:5003/predict', files={"file": ("filename", open(image_detect_path, "rb"), "image/jpeg")}).json()
        vehicle_boxes = detect_response['vehicle_boxes']
        vehicle_scores = detect_response['vehicle_scores']
        vehicle_classes = detect_response['vehicle_classes']

        image, detections = run_detection(frame, vehicle_boxes, vehicle_scores, vehicle_classes, net, output_layers, classes, ENCODER, cfg)
        # tracking
        image = draw_tracking(image, TRACKER, detections)

        cv2.imshow('detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_video('images/cam_02.mp4', net, output_layers, cfg, classes)