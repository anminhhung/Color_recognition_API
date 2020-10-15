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

SERVICE_URL = cfg.SERVICE.SERVICE_URL
BACKUP = cfg.SERVICE.BACKUP_DIR
LOG_PATH = cfg.SERVICE.LOG_PATH
RCODE = cfg.RCODE

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

            # call service
            response = requests.post(SERVICE_URL, files={"file": (img_name, open(img_path, "rb"), "image/jpeg")}).json()

            visual_path = response['visual_path']
            vehicles = response['vehicles']


            json_name = visual_path.split('/')[-1]
            json_name = json_name.split('.')[0] + '.json'
            json_path = os.path.join('json_dir', json_name)

            # create result 
            result = {"code": "1000", "visual_path": visual_path, "json_path":json_path, "vehicles": vehicles}

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
    app.run(debug=True, host='0.0.0.0', port=5555)