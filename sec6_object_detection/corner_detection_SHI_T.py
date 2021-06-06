

import cv2
import numpy as np
import matplotlib.pyplot as plt 


# Shi-Tomasi corner detection


img_flat_chess = cv2.imread('flat_chessboard.png')
img_flat_chess = cv2.cvtColor(img_flat_chess, cv2.COLOR_BGR2RGB)
img_flat_chess_gray = cv2.cvtColor(img_flat_chess, cv2.COLOR_BGR2GRAY)


img_real_chess = cv2.imread('real_chessboard.jpg')
img_real_chess = cv2.cvtColor(img_real_chess, cv2.COLOR_BGR2RGB)
img_real_chess_gray = cv2.cvtColor(img_real_chess, cv2.COLOR_BGR2GRAY)

# plt.imshow(img_flat_chess)
# plt.title('flat_chess')
# plt.show()

# plt.imshow(img_flat_chess_gray, cmap='gray')
# plt.title('flat_chess_gray')
# plt.show()

# plt.imshow(img_real_chess_gray, cmap='gray')
# plt.title('real_chess_gray')
# plt.show()
#############################################################################################


# corner = cv2.goodFeaturesToTrack(img_flat_chess_gray, maxCorners=-1, qualityLevel=0.01, minDistance=10)
corner = cv2.goodFeaturesToTrack(img_flat_chess_gray, maxCorners=5, qualityLevel=0.01, minDistance=10)
# corner = cv2.goodFeaturesToTrack(img_flat_chess_gray, maxCorners=64, qualityLevel=0.01, minDistance=10)


""" draw citcles """ 
corners = np.int0(corner)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img_flat_chess, center=(x,y), radius=3, color=(255,0,0), thickness=-1)

plt.imshow(img_flat_chess)
plt.show()




corner = cv2.goodFeaturesToTrack(img_real_chess_gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
corners = np.int0(corner)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img_real_chess, center=(x,y), radius=3, color=(255,0,0), thickness=-1)

plt.imshow(img_real_chess)
plt.show()














