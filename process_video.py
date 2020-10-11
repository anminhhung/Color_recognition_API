import cv2 
import os 
import requests
import time
import random
from time import gmtime, strftime

if __name__ == '__main__':
    fps = 0.0
    URL = 'http://127.0.0.1:5050/predict'

    cap = cv2.VideoCapture('images/cam_02.mp4')
    while True:
        ret, frame = cap.read() 
        
        t1 = time.time() 

        frame = cv2.resize(frame, (400, 400))

        # gen name
        my_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        number = str(random.randint(0, 10000))
        img_name = my_time + '_' + number + '.jpg'
        img_path = os.path.join('backup', img_name)
        cv2.imwrite(img_path, frame)

        response = requests.post(URL, files={"file": (img_name, open(img_path, "rb"), "image/jpeg")}).json()
        print("response: ", response)

        fps = (fps + (1./(time.time()-t1))) / 2
        print("FPS = %f" % (fps))

        cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()   
    cv2.destroyAllWindows()