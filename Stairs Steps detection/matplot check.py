import matplotlib.pylab as plt
import cv2
import numpy as np

image = cv2.imread('7.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

print(image.shape)
height = image.shape[0]
width = image.shape[1]

plt.imshow(image)
plt.show()

