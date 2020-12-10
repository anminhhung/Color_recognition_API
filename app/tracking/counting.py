import cv2 
from shapely.geometry import Point, Polygon

'''
    Check obj in polygon
    input: list vertices
    output: True if in polygon, otherwise
'''
def check_in_polygon(center_point, polygon):
    pts = Point(center_point[0], center_point[1])
    if polygon.contains(pts):
        return True
    
    return False