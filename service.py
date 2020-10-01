from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np 
import time
import logging
import traceback
import os
import tensorflow as tf
import keras 
from keras.models import load_model

from utils.parser import get_config
from src.test_recognize import recognize

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

# create app
app = Flask(__name__)
CORS(app)

@app.route('/predict-color', methods=['POST'])
def predict_color():
    if request.method ==  'POST':
        try:
            file = request.files['file']
            image_file = file.read()
            image = cv2.imdecode(np.frombuffer(image_file, dtype=np.uint8), -1)

            my_color = recognize(image, MODEL, COLOR_LABELS)
            result = {"color": my_color}

            return jsonify(result)
        except Exception as e:
            logger.error(str(e))
            logger.error(str(traceback.print_exc()))
            result = {'code': '1001', 'status': RCODE.code_1001}
        finally:
            return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False, host=HOST, port=PORT)