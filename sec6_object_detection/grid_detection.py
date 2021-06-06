
# grid should be specific made to look like some sort of checkerboard or chessboard type image

import cv2
import numpy as np
import matplotlib.pyplot as plt 


img_flat_chess = cv2.imread('flat_chessboard.png')
# print(type(img_flat_chess))
# plt.imshow(img_flat_chess)
# plt.show()



""" cv2.findChessboardCorners(src, patternSize=, corners=, plt.flag=) """
# must be chessboard type image 
# return a tuple, found -> boolen 
found, corners = cv2.findChessboardCorners(img_flat_chess, patternSize=(7,7))
# print(found)            # True
# print(corners)            # lisr of coordinate 
# # [[ 43.500004  43.500004]]
# #  [[ 87.5       43.5     ]]
# #  [[131.5       43.5     ]]
# #  [[175.5       43.5     ]]
# #  [[219.5       43.5     ]]
# #  [[263.5       43.5     ]]
# #  [[307.5       43.5     ]]
# #  [[ 43.499996  87.50001 ]]
# #  [[ 87.5       87.5     ]]
# #     ... ... .. 
# #  [[219.5      307.5     ]]
# #  [[263.5      307.5     ]]
# #  [[307.5      307.5     ]]]



""" cv2.drawChessboardCorners(src, ) """
# cv2.findChessboardCorners() will pass in found, corners into cv2.drawChessboardCorners()
cv2.drawChessboardCorners(img_flat_chess, patternSize=(7,7),corners=corners, patternWasFound=found)
plt.imshow(img_flat_chess)
# plt.show()          # mark with corner, different color for each row 

##########################################################################################################

img_dot = cv2.imread('dot_grid.png')
# plt.imshow(img_dot)
# plt.show()

""" 
cv2.findCirclesGrid()
cv2.CALIB_CB_SYMMETRIC_GRID - parameter
"""
found, corners = cv2.findCirclesGrid(img_dot, (10,10), cv2.CALIB_CB_SYMMETRIC_GRID)
# print(found)            # True
# print(corners)
# # [[[ 29.5  29.5]]
# #  [[ 89.5  29.5]]
# #  [[149.5  29.5]]
# #  [[209.5  29.5]]
# #  [[269.5  29.5]]
# #  [[329.5  29.5]]
# #     ... ... 
# #  [[389.5 569.5]]
# #  [[449.5 569.5]]
# #  [[509.5 569.5]]
# #  [[569.5 569.5]]]

cv2.drawChessboardCorners(img_dot, patternSize=(10,10),corners=corners, patternWasFound=found)
# print(corners)
# plt.imshow(img_dot)
# plt.show()












