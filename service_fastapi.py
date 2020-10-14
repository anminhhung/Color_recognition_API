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

# file video
VIDEO_FILE = "images/cam_02.mp4"

# create logging
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
logging.basicConfig(filename=os.path.join(LOG_PATH, str(time.time())+".log"), filemode="w", level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_vehicle():
    if request.method ==  'POST':
        try:
            try:
                file = request.files['file']
                img_path = os.path.join(BACKUP, file.filename)

                # detect
                detect_response = requests.post(DETECT_URL, files={"file": ("filename", open(img_path, "rb"), "image/jpeg")}).json()
            except Exception as e:
                print(str(e))
                print(str(traceback.print_exc()))
                result = {'code': '609', 'status': RCODE.code_609}

                return jsonify(result)

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
            
            json_name = visual_path.split('/')[-1]
            json_name = json_name.solit('.')[0] + '.json'
            json_path = os.path.join('json_dir', json_name)

            with open(json_path, 'w') as f:
                output_json = json.dumps(result)
                f.write(output_json)

            return jsonify(result)

        except Exception as e:
            logger.error(str(e))
            logger.error(str(traceback.print_exc()))
            result = {'code': '1001', 'status': RCODE.code_1001}

            return jsonify(result)

# @app.route('/stream')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(get_frame(VIDEO_FILE, cfg.SERVICE.SERVICE_URL),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5050)