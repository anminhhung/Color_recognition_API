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

app = Flask(__name__, template_folder='templates', static_folder='static')

network, class_names, class_colors = darknet.load_network(
        "yolov4_416.cfg",
        "traffic.data",
        "yolov4_416_last.weights",
        batch_size=8
    )

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['file']
            image_file = file.read()
            image = cv2.imdecode(np.frombuffer(image_file, dtype=np.uint8), -1)

            list_bboxes, list_scores, list_labels = predict_yolov4(image, network, class_names, class_colors)
            return_result = {"code": "1000", "vehicle_boxes": list_bboxes, "vehicle_scores": list_scores, "vehicle_classes": list_labels}
        except Exception as e:
            return_result = {'code': '1001', 'status': "unknow error"}
        finally:
            return jsonify(return_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)