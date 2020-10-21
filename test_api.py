import requests

DETECT_URL: 'http://192.168.28.75:5003/predict'
COLOR_URL: 'http://192.168.28.75:5001/predict'
CAR_RECOG_URL: 'http://192.168.28.75:5002/predict'
SERVICE_URL: 'http://192.168.28.75:5555/predict'
image_path = "images/demo.jpg"
image_detect_path = "images/vehicle_detect.png"

def request_color_api(URL, path):
    color_response = requests.post(URL, files={"file": ("filename", open(path, "rb"), "image/jpeg")}).json()
    print(color_response)

def request_car_recog_api(URL, path):
    car_response = requests.post(URL, files={"file": ("filename", open(path, "rb"), "image/jpeg")}).json()
    print(car_response)

def request_vehicles_detection_api(URL, path):
    vehicles_response = requests.post(URL, files={"file": ("filename", open(path, "rb"), "image/jpeg")}).json()
    print(vehicles_response)

def request_full_service(URL, path):
    response = requests.post(URL, files={"file": ("filename", open(path, "rb"), "image/jpeg")}).json()
    print(response)

if __name__ == '__main__':
    # test color api
    # request_color_api('http://192.168.28.75:5001/predict', image_path)
    # request_car_recog_api('http://192.168.28.75:5002/predict', image_path)
    # request_vehicles_detection_api('http://192.168.28.75:5003/predict', image_detect_path)
    # request_full_service('http://192.168.28.75:5555/predict', image_detect_path)
    request_vehicles_detection_api('http://0.0.0.0:5003/predict', image_detect_path)