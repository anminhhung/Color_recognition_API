import requests
import base64
import cv2
import json

URL = "http://127.0.0.1:8000/predict"
IMG_PATH = "images/demo.jpg"

r = requests.post(URL, files={"file": ("filename", open(IMG_PATH, "rb"), "image/jpeg")})

print("respose: ", r.json())
