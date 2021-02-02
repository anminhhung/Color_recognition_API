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
    color_list = [(255, 0, 255), (255, 100, 0), (0, 255, 0), (139, 69, 19), (132, 112, 255), (0, 154, 205), (0, 255, 127), (238, 180, 180),
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
        moi_startX.append(i[0][0])
        moi_startY.append(i[0][1])
        moi_endX.append(i[1][0])
        moi_endY.append(i[1][1])

    for i in range(len(moi_startX)):
        cv2.arrowedLine(img, (moi_startX[i], moi_startY[i]), (
            moi_endX[i], moi_endY[i]), color_list[i], thickness=2, tipLength=0.03)

    return img


def load_class_names(filename):
    with open(filename, 'r', encoding='utf8') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def get_frame(video_file, URL):
    camera = cv2.VideoCapture(video_file)

    while True:
        retval, im = camera.read()

        # gen name
        my_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        number = str(random.randint(0, 10000))
        img_name = my_time + '_' + number + '.jpg'
        img_path = os.path.join('backup', img_name)
        cv2.imwrite(img_path, im)

        response = requests.post(URL, files={"file": (
            img_name, open(img_path, "rb"), "image/jpeg")}).json()

        image_path = response['visual_path']
        image = cv2.imread(image_path)

        imgencode = cv2.imencode('.jpg', image)[1]

        stringData = imgencode.tostring()

        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

    del(camera)


def get_image(image_path):
    # image_path = os.path.join("backup", filename)
    # image = cv2.imread(image_path)
    while True:
        image = cv2.imread(image_path)
        imgencode = cv2.imencode('.jpg', image)[1]
        stringData = imgencode.tostring()

        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')


def get_image_tracking(image_path):
    # image_path = os.path.join("backup", filename)
    # image = cv2.imread(image_path)
    while True:
        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (480, 270))
            imgencode = cv2.imencode('.jpg', image)[1]
            stringData = imgencode.tostring()

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
            imgencode = cv2.imencode('.jpg', image)[1]
            stringData = imgencode.tostring()

            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
        except:
            pass


def create_white_image_with_text(class_name='Unknown', type_name='Unknown', color_name='Unknown',\
                                 moi='Unknown', height=140, width=250):
    white_image = np.zeros((height, width, 3), dtype=np.uint8)
    height, width, c = white_image.shape
    for i in range(height):
        for j in range(width):
            white_image[i][j][0] = 255
            white_image[i][j][1] = 255
            white_image[i][j][2] = 255

    cv2.putText(white_image, 'Class: ' + class_name, (2, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(white_image, 'Type: ' + type_name, (2, 65), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(white_image, 'Color: ' + color_name, (2, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(white_image, 'MOI: ' + moi, (2, 130), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)

    return white_image


def horizontal_merge(image, info_image, width=140, height=250):
    image = cv2.resize(image, (height, width))
    info_image = cv2.resize(info_image, (height, width))
    result = cv2.hconcat([image, info_image])

    return result


def vertical_merge(image, info_image, width=140, height=250):
    '''
      input: image, info_image (image with information), size
      output: merge_image (vertical)
    '''
    image = cv2.resize(image, (height, width))
    info_image = cv2.resize(info_image, (height, width))
    result = cv2.vconcat([image, info_image])

    return result

def predict_color(model, image, crop_size=50):
    height, width = image.shape[:2]
    x_center = int(width/2)
    y_center = int(height/2)
    x_min = x_center - crop_size
    y_min = y_center - crop_size
    x_max = x_center + crop_size
    y_max = y_center + crop_size
    crop_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    b, g, r = cv2.split(crop_image)

    b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
    r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])

    elem_b = numpy.argmax(b_hist)
    elem_g = numpy.argmax(g_hist)
    elem_r = numpy.argmax(r_hist)

    elem_array = numpy.array([[elem_r, elem_g, elem_b]])
    y_pred = model.predict(elem_array)

    return y_pred