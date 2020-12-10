import cv2 
import numpy as np 
import os 

from shapely.geometry import Point, Polygon

def draw_ROI(img, moi, roi_split_region):
    color_list = [(255,0,255), (255,100,0), (0,255,0), (139, 69, 19), (132, 112, 255), (0, 154, 205), (0, 255, 127), (238, 180, 180),
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
        moi_startX.append (i[0][0])
        moi_startY.append (i[0][1])
        moi_endX.append (i[1][0])
        moi_endY.append (i[1][1])
    
    for i in range (len(moi_startX)):
        cv2.arrowedLine(img, (moi_startX[i], moi_startY[i]), (moi_endX[i], moi_endY[i]), color_list[i], thickness=2, tipLength=0.03)

    return img 

if __name__ == '__main__':
    image_path  = 'output_tracking/video_frame.jpg'
    image = cv2.imread(image_path)
    image = draw_ROI(image)

    cv2.imwrite('image.jpg', image)