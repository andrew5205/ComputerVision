
import numpy as np
import matplotlib.pyplot as plt 

import cv2

# regular img -> gray -> convert into binary 
# https://en.wikipedia.org/wiki/Thresholding_(image_processing)

# https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

# https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57

"""Threshold Types"""
# https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576



img_rainbow = cv2.imread('rainbow.jpg')
plt.imshow(img_rainbow)
# print(img_rainbow.shape)                # (550, 413, 3)
# plt.show()


""" Read img gray 
cv2.imread(src, flag=imread_color)
""" 
img_rainbow_gray_0 = cv2.imread('rainbow.jpg',0)
plt.imshow(img_rainbow_gray_0, cmap='gray')
# print(img_rainbow_gray_0.shape)                # (550, 413, 3)
# plt.show()
######################################################################################

"""Threshold Types"""
# https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576


""" Thershold Binary """ 
print(img_rainbow_gray_0.max())             # 214

# cv2.threshold(src, thresh, maxval) => 0 <- TH=127 -> max
# cv2.threshold() - return as tuple - retval, dst 
ret, thresh1 = cv2.threshold(img_rainbow_gray_0, 127, 255, cv2.THRESH_BINARY)

print(ret)              # 127.0
print(thresh1)
# [[  0   0   0 ... 255 255 255]
#  [  0   0   0 ... 255 255 255]
#  [  0   0   0 ... 255 255 255]
#  ...
#  [  0   0   0 ...   0   0   0]
#  [  0   0   0 ...   0   0   0]
#  [  0   0   0 ...   0   0   0]]

# # plt.imshow(thresh1)
plt.imshow(thresh1, cmap='gray')
plt.show()


""" Thershold Binary Invert """ 
ret, thresh2 = cv2.threshold(img_rainbow_gray_0, 127, 255, cv2.THRESH_BINARY_INV)
print(ret)              # 127.0
print(thresh2)
# plt.imshow(thresh2)
plt.imshow(thresh2, cmap='gray')
plt.show()



""" Thershold Truncktion """ 
ret, thresh3 = cv2.threshold(img_rainbow_gray_0, 127, 255, cv2.THRESH_TRUNC)
print(ret)              # 127.0
print(thresh3)
# plt.imshow(thresh3)
plt.imshow(thresh3, cmap='gray')
plt.show()



""" Thershold To ZERO """ 
ret, thresh4 = cv2.threshold(img_rainbow_gray_0, 127, 255, cv2.THRESH_TOZERO)
print(ret)              # 127.0
print(thresh4)
# plt.imshow(thresh4)
plt.imshow(thresh4, cmap='gray')
plt.show()



""" Thershold To ZERO Invert """ 
ret, thresh5 = cv2.threshold(img_rainbow_gray_0, 127, 255, cv2.THRESH_TOZERO_INV)
print(ret)              # 127.0
print(thresh5)
# plt.imshow(thresh5)
plt.imshow(thresh5, cmap='gray')
plt.show()





