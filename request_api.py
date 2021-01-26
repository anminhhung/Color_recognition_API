import requests

image_detect_path = "000819.jpg"

# detect_url = "http://service.aiclub.cs.uit.edu.vn/vehicles_predict_yolov4/predict"
detect_url = "http://192.168.20.151:7003/detect_traffic"

detect_response = requests.post(detect_url, files={"file": ("filename", open(image_detect_path, "rb"), "image/jpeg")}).json()
vehicle_boxes = detect_response['vehicle_boxes']
vehicle_scores = detect_response['vehicle_scores']
vehicle_classes = detect_response['vehicle_classes']

print(vehicle_classes)