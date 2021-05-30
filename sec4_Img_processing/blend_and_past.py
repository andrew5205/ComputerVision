import matplotlib.pyplot as plt 
import cv2
import numpy as np

from numpy.lib.twodim_base import mask_indices 


""" Blending """
# addWeighted()
# new_pixel = A x pixel_1 + B x pixel_2 + Y 



img_1 = cv2.imread('dog_backpack.png')
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
print(img_1.shape)              # (1401, 934, 3)

img_2 = cv2.imread('watermark_no_copy.png')
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
print(img_2.shape)              # (1280, 1277, 3)




""" blending into same size """
img_resize_1 = cv2.resize(img_1,(1200, 1200))
img_resize_2 = cv2.resize(img_2,(1200, 1200))

# plt.imshow(img_resize_1)
# plt.show()
# plt.imshow(img_resize_2)
# plt.show()


""" addWeighted() - need to be the same size """
# new_pixel = A x pixel_1 + B x pixel_2 + Y 
img_blended = cv2.addWeighted(src1=img_resize_1, alpha=0.5, src2=img_resize_2, beta=0.5, gamma=0)
plt.imshow(img_blended)
# plt.show()




######################################################################################################################
""" 
overlay small image on top of a larger image (no blending) 
Numpy re-assignment 
""" 

img_1 = cv2.imread('dog_backpack.png')
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
print(img_1.shape)              # (1401, 934, 3)

img_2 = cv2.imread('watermark_no_copy.png')
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
print(img_2.shape)              # (1280, 1277, 3)


img_small_img_2 = cv2.resize(img_2, (600,600))
plt.imshow(img_small_img_2)
# plt.show()



img_large = img_1
img_small = img_small_img_2

x_offset = 0 
y_offset = 0 

x_end = x_offset + img_small.shape[1]
y_end = y_offset + img_small.shape[0]

# print(img_small.shape)          # (600, 600, 3)

# cv2 thinks in (y,x)
img_large[y_offset:y_end, x_offset:x_end] = img_small
# img_large[0:600, 0:600] = img_small

plt.imshow(img_large)
# plt.show()






######################################################################################################################
""" 
Blend together images of different size
"""

img_1 = cv2.imread('dog_backpack.png')
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
print(img_1.shape)              # (1401, 934, 3)

img_2 = cv2.imread('watermark_no_copy.png')
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
print(img_2.shape)              # (1280, 1277, 3)


img_small_img_2 = cv2.resize(img_2, (600,600))
plt.imshow(img_small_img_2)
# plt.show()


# print(img_1.shape)              # (1401, 934, 3)

x_offset_b = 934 - 600 
y_offset_b = 1401 - 600 


# print(img_small_img_2.shape)        # (600, 600, 3)
row, col, channels = img_small_img_2.shape

# print(row)      # 600 
# print(col)      # 600

img_roi = img_1[y_offset_b:1401,x_offset_b:943]
plt.imshow(img_roi)
# plt.show()


""" Create mask """
img_2_gray = cv2.cvtColor(img_small_img_2, cv2.COLOR_RGB2GRAY)
plt.imshow(img_2_gray, cmap="gray")
# plt.show()

""" color inverse - cv2.bitwise_not() 

bitwise_not(src[, dst[, mask]]) -> dst

@brief Inverts every bit of an array.

The function cv::bitwise_not calculates per-element bit-wise inversion 
of the input array: \f[\texttt{dst} (I) = \neg \texttt{src} (I)\f] 

In case of a floating-point input array, its machine-specific bit representation (usually IEEE754-compliant) 
is used for the operation. 

In case of multi-channel arrays, each channel is processed independently.
"""
img_mask_inv = cv2.bitwise_not(img_2_gray)
plt.imshow(img_mask_inv)
plt.imshow(img_mask_inv, cmap='gray')
# plt.show()
# print('###########')
# # inversed will left only one channel 
print(img_mask_inv.shape)               # (600, 600)


""" creat a new img by np.full() """
# Return a new array of given shape and type, filled with fill_value.
img_white_background = np.full(img_small_img_2.shape, 255, dtype=np.uint8)
print(img_white_background.shape)               # (600, 600, 3)
print(img_white_background)
# [[255 255 255]
#   [255 255 255]
#   [255 255 255]
#   ...
#   [255 255 255]
#   [255 255 255]
#   [255 255 255]]]



""" cv2.bitwise_or() -
src1 first input array or a scalar.

bitwise_or(src1, src2[, dst[, mask]]) -> dst
"""
img_bk = cv2.bitwise_or(img_white_background, img_white_background, mask=img_mask_inv)
print(img_bk.shape)             # (600, 600, 3)
plt.imshow(img_bk)
# plt.show()


# original img_small_img_2 in RED
img_fg = cv2.bitwise_or(img_small_img_2, img_small_img_2, mask=img_mask_inv)
print(img_fg.shape)             # (600, 600, 3)
plt.imshow(img_fg)
# plt.show()



img_final_roi = cv2.bitwise_or(img_roi, img_fg)
plt.imshow(img_final_roi)
# plt.show()




img_large_ready_to_use = img_1
img_small_ready_to_use = img_final_roi

img_large_ready_to_use[y_offset_b:y_offset_b+img_small_ready_to_use.shape[0], x_offset_b:x_offset_b+img_small_ready_to_use.shape[1]] = img_small_ready_to_use
plt.imshow(img_large_ready_to_use)
plt.show()



