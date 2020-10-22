from __future__ import division, print_function, absolute_import
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse

from libs import preprocessing
from libs import nn_matching
from libs import Detection
from libs import Tracker
from utils import generate_detections as gdet
from libs import Detection as ddet
from collections import deque

from src import detect

def run_detection(image, list_boxes, list_scores, list_classes, encoder, cfg):
    # detections_in_ROI = []

    features = encoder(image, list_boxes)
    detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                    zip(list_boxes, list_scores, list_classes, features)]

    # Run non-max suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, cfg.DEEPSORT.NMS_MAX_OVERLAP, scores)
    detections = [detections[i] for i in indices]

    # for det in detections:
    #     bbox = det.to_tlbr()
    #     centroid_det = (int((bbox[0] + bbox[2])//2), int((bbox[1] + bbox[3])//2))
        # if check_in_polygon(centroid_det, self.polygon_ROI):
        #     detections_in_ROI.append(det)

    for det in detections:
        bbox = det.to_tlbr()
        score = "%.2f" % round(det.confidence * 100, 2) + "%"
        # cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(
        #     bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        # if len(classes) > 0:
        cls = det.cls
        cv2.putText(image, str(cls) + ": " + score, (int(bbox[0]), int(bbox[3])), 0,
                    1e-3 * image.shape[0], (0, 255, 0), 1)

    return image, detections

def draw_tracking(image, tracker, detections):
    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        
        bbox = track.to_tlbr()
        centroid = (int((bbox[0] + bbox[2])//2), int((bbox[1] + bbox[3])//2))

        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 1)
        cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # draw track line
        image = track.draw_track_line(image)

    image_path = "output_tracking/video_frame.jpg"
    cv2.imwrite(image_path, image)

    return image_path
