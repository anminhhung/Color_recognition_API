import cv2 
import os 
import requests
import time
import random
import numpy
import sys
from time import gmtime, strftime

from flask import Flask, render_template, Response

app = Flask(__name__)
video_file = "images/cam_02.mp4"
URL = 'http://127.0.0.1:5050/predict'

@app.route('/')
def index():
    return render_template('index.html')

def get_frame():
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
        # with open("demo.txt", "a+") as f:
        #     f.write("{}\n".format(response))

        imgencode=cv2.imencode('.jpg',im)[1]
        
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

    del(camera)

@app.route('/calc')
def calc():
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True, threaded=True)


# if __name__ == '__main__':
#     fps = 0.0
#     URL = 'http://127.0.0.1:5050/predict'

#     cap = cv2.VideoCapture('images/cam_02.mp4')
#     while True:
#         ret, frame = cap.read() 
        
#         t1 = time.time() 

#         frame = cv2.resize(frame, (400, 400))

#         # gen name
#         my_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
#         number = str(random.randint(0, 10000))
#         img_name = my_time + '_' + number + '.jpg'
#         img_path = os.path.join('backup', img_name)
#         cv2.imwrite(img_path, frame)

#         response = requests.post(URL, files={"file": (img_name, open(img_path, "rb"), "image/jpeg")}).json()
#         print("response: ", response)

#         fps = (fps + (1./(time.time()-t1))) / 2
#         print("FPS = %f" % (fps))

#         cv2.imshow('video', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()   
#     cv2.destroyAllWindows()