import numpy as np 
import cv2 
import os 
import logging
import time
import traceback

from utils.parser import get_config

from libs import preprocessing
from libs import nn_matching
from libs import Detection
from libs import Tracker
from utils import generate_detections as gdet
from libs import Detection as ddet
from collections import deque

from app import app

# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')

HOST = cfg.SERVICE.SERVICE_IP
PORT = cfg.SERVICE.TRACKING_PORT

if __name__ == '__main__':            
    app.run(debug=True, host=HOST, port=PORT)