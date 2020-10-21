import cv2
import numpy as np
import os
import random

net = cv2.dnn.readNet("models/yolov4.weights", "models/yolov4.cfg")

# Name custom object
classes = ['Loai1', 'Loai2', 'Loai3', 'Loai4', 'Loai5']
# classes = load_class_names('models/vehicle_classes.txt')

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def detect(img, net, output_layers, classes):
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(str(classes[class_id]))
                print("class: ", classes[class_id])

    return boxes, confidences, class_ids

def detect_video(video_path, classes):
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    # start detect video
    cap = cv2.VideoCapture(video_path)
    while (True):
        ret, frame = cap.read()
        
        # predict
        boxes, confidences, class_ids = detect(frame, net, output_layers, classes)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                # label = str(classes[class_ids[i]])
                # color = colors[class_ids[i]]
                color = colors[1]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        cv2.imshow('detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_video('images/cam_02.mp4')
    # image, image_path = detect_potHole(cv2.imread("images/img-3.jpg"))
    # cv2.imshow("demo", image)
    # cv2.waitKey(0)