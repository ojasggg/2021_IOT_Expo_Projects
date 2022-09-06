import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image

I = image.imread('5.jpg')
blur = cv2.GaussianBlur(I, (5, 5), 0, 0)


# Canny Edge Detection:
Threshold1 = 100
Threshold2 = 350
apertureSize = 3
E = cv2.Canny(I, Threshold1, Threshold2,apertureSize)

G = cv2.cvtColor(E, cv2.COLOR_GRAY2BGR)
H = cv2.cvtColor(E, cv2.COLOR_GRAY2BGR)

y_keeper_for_lines = []



# cv2.imshow("image", E)
# cv2.waitKey(0)
Rres = 1
Thetares = 1*np.pi/180
Threshold = 1
minLineLength = 5
maxLineGap = 100
lines = cv2.HoughLinesP(E,rho = 1,theta = 1*np.pi/180,threshold = 300,minLineLength = 100,maxLineGap =50 )

i = 0
for a in range(i,i+1):
    if a<len(lines):
        l = lines[a]
        cv2.line(H,(l[0],l[1]),(l[2],l[3]),(255,0,0),5, cv2.LINE_AA)

l = lines[0]
cv2.line(G,(0,l[1]),(I.cols,l[1]),(255,0,0),5, cv2.LINE_AA)
y_keeper_for_lines.append(l[1])

okey = 1
counter = 1

j=1
for b in range(j,j+1):
    if j<len(lines):
        l = lines[j]
        for m in y_keeper_for_lines:
            if (abs(m-l[1])<15):
                okey = 0
        if okey == True:
            cv2.line(G, (0, l[1]), (I.cols, l[1]), (255, 0, 0), 5, cv2.LINE_AA)
            y_keeper_for_lines.append(l[1])
            counter = counter+1
        okey = 1
# N = lines.shape[0]
# for i in range(N):
#     x1 = lines[i][0][0]
#     y1 = lines[i][0][1]
#     x2 = lines[i][0][2]
#     y2 = lines[i][0][3]
#     cv2.line(G,(x1,y1),(x2,y2),(255,0,0),5)

print(counter)
plt.figure(),plt.imshow(G),plt.title('Hough Lines'),plt.axis('off')
plt.show()