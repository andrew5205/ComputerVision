
# morphological operator are sets of kernels that can achieve a variety of effects, such as reducing noise

import numpy as np 
import matplotlib.pyplot as plt 
import cv2
from numpy.core.fromnumeric import size
from numpy.lib.function_base import gradient 



def load_img():
    img_blank = np.zeros((600,600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_blank, text='ABCDE', org=(50, 300), fontFace=font, fontScale=5, color=(255,255,255), thickness=20)
    return img_blank


def display_img(img):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


img = load_img()
# display_img(img)


my_kernel = np.ones((5,5), dtype=np.uint8)
# print(my_kernel)
# # [[1 1 1 1 1]
# #  [1 1 1 1 1]
# #  [1 1 1 1 1]
# #  [1 1 1 1 1]
# #  [1 1 1 1 1]]


""" erosion - cv2.erode() 
iteration is the key here 
""" 
result = cv2.erode(img, my_kernel, iterations=1)
# result = cv2.erode(img, kernel, iterations=4)
# display_img(result)
###################################################################################################################




""" white noise creation - np.random.randint() """ 
# high val is int, (exclusive) 
white_noise = np.random.randint(low=0, high=2, size=(600,600))
# print(white_noise)
# # [[0 0 0 ... 0 1 0]
# #  [1 1 0 ... 0 0 1]
# #  [1 0 1 ... 0 0 1]
# #  ...
# #  [1 0 1 ... 1 1 0]
# #  [1 0 0 ... 0 0 1]
# #  [0 1 0 ... 0 1 0]]
# display_img(white_noise)


# to match the white_noise 1 to 255, match same scale as base image img_blank
white_noise = white_noise * 255 
# display_img(white_noise)


img_blank_with_white_noise = white_noise + img 
# display_img(img_blank_with_white_noise)



""" Opening - good for removing background noise """ 
img_opening = cv2.morphologyEx(img_blank_with_white_noise, cv2.MORPH_OPEN, kernel=my_kernel)
# display_img(img_opening)




""" """
img = load_img()
# high val is int, (exclusive) 
black_noise = np.random.randint(low=0, high=2, size=(600, 600))
black_noise = black_noise * -255 
# print(black_noise)
# # [[   0 -255 -255 ... -255    0    0]
# #  [   0    0 -255 ...    0    0    0]
# #  [-255    0    0 ... -255 -255 -255]
# #  ...
# #  [   0 -255 -255 ... -255    0    0]
# #  [   0    0 -255 ...    0    0 -255]
# #  [   0    0    0 ... -255 -255 -255]]


img_blank_with_black_noise = black_noise + img 
img_blank_with_black_noise[img_blank_with_black_noise == -255] = 0

# display_img(img_blank_with_black_noise)


""" Closing - good for removing foreground noise """ 
img_closing = cv2.morphologyEx(img_blank_with_black_noise, cv2.MORPH_CLOSE, kernel=my_kernel)
# display_img(img_closing)
###################################################################################################################




# reset img
img = load_img()
# display_img(img)

""" cv2.MORPH_GRAIDENT() - good for edge dection  """ 
img_gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel=my_kernel)
display_img(img_gradient)


