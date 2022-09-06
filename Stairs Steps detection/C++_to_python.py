import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

I = image.imread('4.jpg')
cv2.imshow("image", I)

blur = cv2.GaussianBlur(I, (5, 5), 0, 0)

# Canny Edge Detection:
Threshold1 = 80
Threshold2 = 240
apertureSize = 3
canny = cv2.Canny(blur, Threshold1, Threshold2,apertureSize)

outimg = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
control = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

lines = cv2.HoughLinesP(canny,rho = 1,theta = 1*np.pi/180,threshold = 30,minLineLength = 40,maxLineGap = 5)

y= []
l=[]
i = 1
for i in range(i,i < lines.size(),1):
    l = lines[i]
    cv2.line(control,(l[0], l[1]),(l[2], l[3]), (0,0,255), 3)

l= lines[0]
cv2.line(control,(l[0], l[1]),(l[2], l[3]), (0,0,255), 3)
y.append(l[1])

okey = 1
stair_counter = 1
# b=1
for x in (i , i < lines.size(), 1 ):

    l = lines[i]
    for m in y:
        if(abs(m-l[1])<15):
            okey = 0


    if okey:
        cv2.line(outimg,(0, l[1]),(I.cols, l[1]),(0,0,255), 3)
        y.append(l[1])
        stair_counter += 1

    okey = 1


plt.title(outimg,"Stair number:" + str(stair_counter),(40,60),1.5,(0,255,0),2)
cv2.imshow("Before", I)
cv2.imshow("Control", control)
cv2.imshow("detected lines", outimg)
cv2.waitKey(0)

