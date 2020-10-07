import os
import cv2
import json
import random
import itertools
import numpy as np
import argparse
import cv2

def predict(image, predictor):
    outputs = predictor(image)

    boxes = outputs['instances'].pred_boxes
    scores = outputs['instances'].scores
    classes = outputs['instances'].pred_classes

    list_boxes = []
    list_scores = []
    list_classes = []

    for i in range(len(classes)):
        if (scores[i] > 0.5):
            for j in boxes[i]:
                x = int(j[0])
                y = int(j[1])
                w = int(j[2]) - x
                h = int(j[3]) - y

            score = float(scores[i])
            class_id = int(classes[i])
            list_boxes.append([x, y, w, h])
            list_scores.append(score)
            list_classes.append(class_id)

    return list_boxes, list_scores, list_classes