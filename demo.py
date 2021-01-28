from utils.utils import create_white_image_with_text, horizontal_merge, vertical_merge
import cv2
import numpy as np

image = cv2.imread('images/demo.jpg')
white_image = create_white_image_with_text()

hor_image = horizontal_merge(image, white_image)
ver_image = vertical_merge(image, white_image)

cv2.imshow("image", ver_image)
cv2.waitKey(0)