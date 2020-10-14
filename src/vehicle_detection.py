import os
import cv2
import json
import random
import itertools
import numpy as np
import argparse
import cv2

from time import gmtime, strftime

def predict(image, predictor, list_labels):
    outputs = predictor(image)

    boxes = outputs['instances'].pred_boxes
    scores = outputs['instances'].scores
    classes = outputs['instances'].pred_classes

    # list_boxes = []
    list_paths = []
    list_scores = []
    list_classes = []

    for i in range(len(classes)):
        if (scores[i] > 0.6):
            for j in boxes[i]:
                x = int(j[0])
                y = int(j[1])
                w = int(j[2]) - x
                h = int(j[3]) - y

            score = float(scores[i])
            class_id = list_labels[int(classes[i])]

            # store vehicle
            vehicle_image = image[y: (y+h), x: (x+w), :]
            time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
            number = str(random.randint(0, 10000))
            img_name = time + '_' + number + '.jpg'
            img_path = os.path.join('output_detect', img_name)
            cv2.imwrite(img_path, vehicle_image)

            # visual box
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)

            # list_boxes.append([x, y, w, h])
            list_paths.append(img_path)
            list_scores.append(score)
            list_classes.append(class_id)
    
    visual_path = os.path.join('visual', strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + '_' + str(random.randint(0, 10000)) + '.jpg')
    cv2.imwrite(visual_path, image)

    return visual_path, list_paths, list_scores, list_classes