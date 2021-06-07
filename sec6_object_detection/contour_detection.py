

import cv2
import numpy as np
import matplotlib.pyplot as plt 



img_int_ext = cv2.imread('internal_external.png', 0)
# print(img_int_ext.shape)            # (652, 1080)

# plt.imshow(img_int_ext, cmap='gray')
# plt.show()



# cv2.findContours()
# cv2.RETR_CCOMP - complete - internal + external 
contours, hierarchy  = cv2.findContours(img_int_ext, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# print(type(contours))           # <class 'list'>
# print(len(contours))            # 22

# print(type(hierarchy))          # <class 'numpy.ndarray'>
# print(hierarchy)
# # [[[ 4 -1  1 -1]
# #   [ 2 -1 -1  0]
# #   [ 3  1 -1  0]
# #   [-1  2 -1  0]
# #   [21  0  5 -1]
# #   [ 6 -1 -1  4]
# #   [ 7  5 -1  4]
# #   [ 8  6 -1  4]
# #   [ 9  7 -1  4]
# #   [10  8 -1  4]
# #   [11  9 -1  4]
# #   [12 10 -1  4]
# #   [13 11 -1  4]
# #   [14 12 -1  4]
# #   [15 13 -1  4]
# #   [16 14 -1  4]
# #   [17 15 -1  4]
# #   [18 16 -1  4]
# #   [19 17 -1  4]
# #   [20 18 -1  4]
# #   [-1 19 -1  4]
# #   [-1  4 -1 -1]]]

#################################################################################################

contour_external = np.zeros(img_int_ext.shape)
# print(contour_external.shape)           # (652, 1080)

for i in range(len(contours)):

    # External 
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(contour_external, contours, i, 255, -1)
        
plt.imshow(contour_external, cmap='gray')
plt.title('External contour')
plt.show()





contour_internal = np.zeros(img_int_ext.shape)
for i in range(len(contours)):

    # internal 
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(contour_internal, contours, i, 255, -1)
        
plt.imshow(contour_internal, cmap='gray')
plt.title('Internal contour')
plt.show()








