
# Gradient: directional change in the intensity or color in image

# edge detection:
#     objection, object tracking, image classification


import numpy as np 
import matplotlib.pyplot as plt 
import cv2


def display_img(img):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


img_sudoku = cv2.imread('sudoku.jpg', 0)                # 0 makes it in gray
# display_img(img_sudoku)


sobel_x = cv2.Sobel(img_sudoku, cv2.CV_64F, 1, 0, ksize=5)
# display_img(sobel_x)



sobel_y = cv2.Sobel(img_sudoku, cv2.CV_64F, 0, 1, ksize=5)
# display_img(sobel_y)



laplacian = cv2.Laplacian(img_sudoku, cv2.CV_64F)
# display_img(laplacian)



blended = cv2.addWeighted(src1=sobel_x, alpha=0.5, src2=sobel_y, beta=0.5, gamma=0)
# display_img(blended)



ret, th1 = cv2.threshold(img_sudoku, 100, 255, cv2.THRESH_BINARY)
# display_img(th1)




kernel = np.ones((4,4), np.uint8)
gradient = cv2.morphologyEx(blended, cv2.MORPH_GRADIENT, kernel)
display_img(gradient)





