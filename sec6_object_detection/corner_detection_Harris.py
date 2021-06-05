
import cv2
import numpy as np
import matplotlib.pyplot as plt 


# Harris corner detection 
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



gray = np.float32(img_flat_chess_gray)
# print(gray)
# # [[197. 197. 197. ... 127. 127. 127.]
# #  [197. 197. 197. ... 127. 127. 127.]
# #  [197. 197. 197. ... 127. 127. 127.]
# #  ...
# #  [127. 127. 127. ... 197. 197. 197.]
# #  [127. 127. 127. ... 197. 197. 197.]
# #  [127. 127. 127. ... 197. 197. 197.]]
#########################################################################################################

""" cv2.cornerHarris() """ 
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)
# print(dst)

""" cv2.dilate() """ 
dst = cv2.dilate(dst, None)
# print(dst)


""" apply Corner Harris formula"""
# from dilated cornerHarris result, if %1 of it's value is greater then max val of deliated cornerHarris
# turn into (255, 0, 0) RGB
img_flat_chess[dst > 0.01 * dst.max()] = [255,0,0]

plt.imshow(img_flat_chess)
plt.show()
##########################################################################################################


gray1 = np.float32(img_real_chess_gray)
dst = cv2.cornerHarris(src=gray1, blockSize=2, ksize=3, k=0.04)

dst = cv2.dilate(dst, None)

img_real_chess[dst > 0.01 * dst.max()] = [255,0,0]
plt.imshow(img_real_chess)
plt.show()






