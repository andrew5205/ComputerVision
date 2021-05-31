# Image kernals
# https://setosa.io/ev/image-kernels/
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
from numpy.lib.function_base import disp


""" function """
def load_img():
    img = cv2.imread('bricks.jpg').astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img)
    # print(type(img))                # <class 'numpy.ndarray'>
    return img 
# load_img()


def display_img(img):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()
    
    
i = load_img()
# display_img(i)



""" 
https://numpy.org/doc/stable/reference/generated/numpy.power.html
"""
# gamma < 1, makes image brighter
gamma_b = 0.25 
# gamma > 1, makes image darker
gamma_d = 8

result = np.power(i, gamma_b)
# # result = np.power(i, gamma_d)
# display_img(result)



""" 2D convolution """
img_brick_org = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img_brick_org, text='bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255,0,0), thickness=5)
# display_img(img_brick_org)



""" make kernel - np.array"""
kernel = np.ones(shape=(5,5), dtype=np.float32) / 25 
# print(kernel)
# # [[0.04 0.04 0.04 0.04 0.04]
# #  [0.04 0.04 0.04 0.04 0.04]
# #  [0.04 0.04 0.04 0.04 0.04]
# #  [0.04 0.04 0.04 0.04 0.04]
# #  [0.04 0.04 0.04 0.04 0.04]]



""" cv2.filter2D(src, dst, ddepth, kernelanchor, delta, borderType) """
# https://stackoverflow.com/questions/43392956/explanation-for-ddepth-parameter-in-cv2-filter2d-opencv
# https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
dst = cv2.filter2D(img_brick_org, -1, kernel)
# display_img(dst)
####################################################################################################################


# reset img 
img_brick_org = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img_brick_org, text='bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255,0,0), thickness=5)
# display_img(img_brick_org)


""" cv2 built-in kernel """
# img_blurred = cv2.blur(img_brick_org, ksize=(5,5))
img_blurred = cv2.blur(img_brick_org, ksize=(50,50))              # bigger kernel size makes distortion -> more blur
# display_img(img_blurred)
####################################################################################################################



# reset img 
img_brick_org = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img_brick_org, text='bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255,0,0), thickness=5)
# display_img(img_brick_org)

""" cv2.GaussianBlur() """
img_gaussianBlur = cv2.GaussianBlur(img_brick_org, ksize=(5,5), sigmaX=5)
# display_img(img_gaussianBlur)
####################################################################################################################



# res_brick_orget img 
img_brick_org = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img_brick_org, text='bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255,0,0), thickness=5)
# display_img(img_brick_org)

""" cv2.medianBlur() """ 
img_medianBlur = cv2.medianBlur(img_brick_org, ksize=5)
# display_img(img_medianBlur)
####################################################################################################################



""" compare w/ noise """ 
img_sammy = cv2.imread('sammy.jpg')
img_sammy = cv2.cvtColor(img_sammy, cv2.COLOR_BGR2RGB)
# display_img(img_sammy)


img_sammy_noise = cv2.imread('sammy_noise.jpg')
# display_img(img_sammy_noise)


img_medianBlur_noise = cv2.medianBlur(img_sammy, ksize=5)
# img_medianBlur_noise = cv2.medianBlur(img_sammy, ksize=11)
# display_img(img_medianBlur_noise)
####################################################################################################################



# reset img 
img_brick_org = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img_brick_org, text='bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255,0,0), thickness=5)
display_img(img_brick_org)


""" cv2.bilateralFilter() """ 
img_bilateralFilter = cv2.bilateralFilter(img_brick_org, 9, 75, 75)
# display_img(img_bilateralFilter)






