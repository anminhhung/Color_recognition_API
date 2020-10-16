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
from utils.utils import load_class_names, get_image

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
HOST = cfg.SERVICE.SERVICE_IP
PORT = cfg.SERVICE.SERVICE_PORT

if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
logging.basicConfig(filename=os.path.join(LOG_PATH, str(time.time())+".log"), filemode="w", level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        try:
            try: 
                file = request.files['file']
                image_file = file.read()
                image = cv2.imdecode(np.frombuffer(image_file, dtype=np.uint8), -1)
            except Exception as e:
                print(str(e))
                print(str(traceback.print_exc()))
                result = {'code': '609', 'status': RCODE.code_609}

                return jsonify(result)

            # save image
            time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
            number = str(random.randint(0, 10000))
            img_name = time + '_' + number + '.jpg'
            img_path = os.path.join(BACKUP, img_name)
            cv2.imwrite(img_path, image)

            # detect
            detect_response = requests.post(DETECT_URL, files={"file": ("filename", open(img_path, "rb"), "image/jpeg")}).json()

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

            json_name = visual_path.split('/')[-1]
            json_name = json_name.split('.')[0] + '.json'
            json_path = os.path.join('json_dir', json_name)

            # create result 
            result = {"code": "1000", "visual_path": visual_path, "json_path":json_path, "vehicles": list_vehicle}

            with open(json_path, 'w') as f:
                output_json = json.dumps(result)
                f.write(output_json)
            
            return jsonify(result)

        except Exception as e:
            logger.error(str(e))
            logger.error(str(traceback.print_exc()))
            result = {'code': '1001', 'status': RCODE.code_1001}

            return jsonify(result)

@app.route('/stream/<path:filename>"')
def stream_image(filename):
    try:
        image_path = os.path.join("backup", filename)
        image = cv2.imread(image_path)
    except Exception as e:
        print(str(e))
        print(str(traceback.print_exc()))
        result = {'code': '609', 'status': RCODE.code_609}

    return Response(get_image(image),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=False, host=HOST, port=PORT)