
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 



def display(img, cmap='gray'):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()



sep_coins = cv2.imread('pennies.jpg')
# display(sep_coins)



# Median Blur
# Grayscale 
# Binary threshold 
# Find Contour



sep_blur = cv2.medianBlur(sep_coins, 25)
# display(sep_blur)

gray_sep_coins = cv2.cvtColor(sep_blur, cv2.COLOR_BGR2GRAY)
# display(gray_sep_coins)


ret, sep_thresh = cv2.threshold(gray_sep_coins, 160, 255, cv2.THRESH_BINARY_INV)
# display(sep_thresh)



countours, hierarchy = cv2.findContours(sep_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(countours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins, countours, i, (255,0,0), 10)
display(sep_coins)



