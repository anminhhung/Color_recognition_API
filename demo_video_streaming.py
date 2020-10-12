from flask import Flask, render_template, Response
import cv2
import sys
import numpy

app = Flask(__name__)
video_file = "images/cam_02.mp4"

@app.route('/')
def index():
    return render_template('index.html')

def get_frame():
    camera=cv2.VideoCapture(video_file)

    while True:
        retval, im = camera.read()
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
