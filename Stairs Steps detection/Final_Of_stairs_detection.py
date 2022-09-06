import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

img = image.imread('5.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

print(img.shape)

# To find out the region of interest vertices.
height = img.shape[0]
width = img.shape[1]
region_of_interest_vertices = [(0,height),(width/2,0),(width,height)]

# Function to crop the image to get region of interest.
def region_of_interest (img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask,vertices,match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



# Copy edges to the images that will display the results in BGR
cdst = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# Canny Edge Detection:
Threshold1 = 100
Threshold2 = 330
apertureSize = 3
canny = cv2.Canny(cdst, Threshold1, Threshold2,apertureSize,None)

cdst1 = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

# This is the region of interest cropped Image
cropped_image = region_of_interest(canny,np.array([region_of_interest_vertices],np.int32))

# Houghline() detection
def draw_line(img,lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0],img.shape[1],3),dtype = np.uint8)

    if lines is not None:
        for i in range(0, len(lines)):
            # rho,theta = lines[0]
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)  # Polar coordinates to cartesian coordinate
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            a = a + 1
            cv2.line(blank_image, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)

    img = cv2.addWeighted(img, 0.8, blank_image, 1,0.0)
    print(len(lines))
    return img


lines = cv2.HoughLines(cropped_image, 1, np.pi/180, 100, None, 0, 0)


# HoughlineP() detection
def draw_lineP(img,lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0],img.shape[1],3),dtype = np.uint8)
    if lines is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(blank_image, (l[0], l[1]), (l[2], l[3]), (255,0,0), 5, cv2.LINE_AA)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    print(len(linesP))
    return img


linesP = cv2.HoughLinesP(cropped_image,rho= 1,theta= np.pi / 180,threshold= 200,lines=np.array([]),minLineLength = 100, maxLineGap = 100)

# If you are using Houghline() method then run this and comment other.
image_with_line = draw_line(img,lines)
plt.imshow(image_with_line)

# OR if using HoughlineP() method then run this and comment other.
# image_with_lineP = draw_lineP(img,linesP)
# plt.imshow(image_with_lineP)

plt.show()

