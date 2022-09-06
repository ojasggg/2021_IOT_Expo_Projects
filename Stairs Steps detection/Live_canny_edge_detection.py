import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

video_capture = cv2.VideoCapture(0)
img_counter = 0

while True:
    #capture frame by frame
    ret,frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.waitKey(1)
    
    
    minLineLength = 800
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # k = cv2.waitKey(1)
    cv2.imshow("image", frame)



# img = cv2.imread('stairs2.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



video_capture.release()
cv2.imwrite('stairs.jpg', frame)