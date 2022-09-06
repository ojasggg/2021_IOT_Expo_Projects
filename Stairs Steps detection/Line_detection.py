import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

img = cv2.imread('7.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

edges = cv2.Canny(img, 100, 350, apertureSize=3)
cv2.imshow("edges", edges)
cv2.waitKey(0)
minLineLength = 60
maxLineGap = 100
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 290, minLineLength, maxLineGap)
for line in lines:
    # for x1, y1, x2, y2 in line:
    x1,y1,x2,y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 5)

cv2.imshow('image',img)
k = cv2.waitKey()
cv2.destroyAllWindows()
