import cv2 
import os 
import requests
import time
import random
import numpy as np
import sys
from time import gmtime, strftime
from shapely.geometry import Point, Polygon
import itertools

LOGO = 'app/static/figure/logo.png'

def draw_ROI(img, moi, roi_split_region):
    color_list = [(255,0,255), (255,100,0), (0,255,0), (139, 69, 19), (132, 112, 255), (0, 154, 205), (0, 255, 127), (238, 180, 180),
                  (0, 100, 0), (238, 106, 167), (221, 160, 221), (0, 128, 128)]

    # moi = [[[549, 144], [297, 505]], [[925, 487], [715, 144]]]
    # moi = cfg.CAM.MOI
    moi_startX = []
    moi_endX = []
    moi_startY = []
    moi_endY = []

    # ROI_SPLIT_REGION = [ [[1,460],[2,336], [150, 210],[599,229],[594,462]], [[594,462],[599,229],[963,247],[1274,464]]]
    # roi_split_region = cfg.CAM.ROI_SPLIT_REGION
    for index, region in enumerate(roi_split_region):
        region = np.array(region)
        cv2.drawContours(img, [region], -1, color_list[index], 2)

    # plot MOI
    # plot MOI
    for i in moi:
        moi_startX.append (i[0][0])
        moi_startY.append (i[0][1])
        moi_endX.append (i[1][0])
        moi_endY.append (i[1][1])
    
    for i in range (len(moi_startX)):
        cv2.arrowedLine(img, (moi_startX[i], moi_startY[i]), (moi_endX[i], moi_endY[i]), color_list[i], thickness=2, tipLength=0.03)

    return img 

def load_class_names(filename):
    with open(filename, 'r', encoding='utf8') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def get_frame(video_file, URL):
    camera=cv2.VideoCapture(video_file)

    while True:
        retval, im = camera.read()

        # gen name
        my_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        number = str(random.randint(0, 10000))
        img_name = my_time + '_' + number + '.jpg'
        img_path = os.path.join('backup', img_name)
        cv2.imwrite(img_path, im)

        response = requests.post(URL, files={"file": (img_name, open(img_path, "rb"), "image/jpeg")}).json()

        image_path = response['visual_path']
        image = cv2.imread(image_path)

        imgencode=cv2.imencode('.jpg',image)[1]
        
        stringData=imgencode.tostring()

        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

    del(camera)

def get_image(image_path):
    # image_path = os.path.join("backup", filename)
    # image = cv2.imread(image_path)
    while True:
        image = cv2.imread(image_path)
        imgencode=cv2.imencode('.jpg',image)[1]
        stringData=imgencode.tostring()

        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

def get_image_tracking(image_path):
    # image_path = os.path.join("backup", filename)
    # image = cv2.imread(image_path)
    while True:
        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (480, 270))
            imgencode=cv2.imencode('.jpg',image)[1]
            stringData=imgencode.tostring()

            yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
        except:
            pass

def get_crop_track1(image_path):
    # image_path = os.path.join("backup", filename)
    # image = cv2.imread(image_path)
    while True: 
        try:
            image = cv2.imread(image_path)
            # print("IMAGE SIZE: ", image.shape)
            imgencode=cv2.imencode('.jpg',image)[1]
            stringData=imgencode.tostring()

            yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
        except:
            pass

def get_class_vehicle():
   for i, c in enumerate(itertools.cycle('\|/-')):
        yield "data: %s %d %d\n\n" % (c, i, int(number))
        time.sleep(.1)  # an artificial delay