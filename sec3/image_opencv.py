import numpy as np 
import matplotlib.pyplot as plt

import cv2
from numpy.lib.ufunclike import fix

"""cv2.imread('path')"""
img = cv2.imread('00-puppy.jpg')
print(type(img))                # <class 'numpy.ndarray'> 


# imgWrongPath = cv2.imread('path/does/not/exist.jpg')
# print(type(imgWrongPath))       # <class 'NoneType'>


print(img.shape)                # (1300, 1950, 3)


plt.imshow(img)
# plt.show()




""" cv2.cvtcolor() """ 
# MATPLOTLIB --> Red Green Blue, RGB
# OPENCV --> Blue Green Red, BGR 

# cv2.cvtcolor(src, cv2.COLOR_trans_code)
fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(fix_img)
# plt.show()


# cv2.IMREAD_convertCode
img_gray = cv2.imread('00-puppy.jpg',cv2.IMREAD_GRAYSCALE)
# plt.imshow(img_gray, cmap='gray')     # matplotlib read gray

print(img_gray.shape)               # (1300, 1950)
print(img_gray)
# [[85 87 88 ... 26 26 26]
#  [85 86 86 ... 26 26 26]
#  [85 84 84 ... 26 26 26]
#  ...
#  [25 26 26 ... 27 28 28]
#  [26 26 25 ... 27 28 28]
#  [26 25 25 ... 27 28 28]]

plt.imshow(img_gray)
# plt.show()



""" cv2.resize() """
img_resize = cv2.resize(fix_img,(1000,500))
plt.imshow(img_resize)
# plt.show()
print(img_resize.shape)             # (500, 1000, 3)


# check doc resize by ratio
img_resize_by_ratio = cv2.resize(fix_img, (0,0), fix_img, 0.5, 0.5)
plt.imshow(img_resize_by_ratio)
# plt.show()
print(img_resize_by_ratio.shape)                # (650, 975, 3)



""" cv2.flip() """
img_flipped = cv2.flip(fix_img,0)           # vertical, upside down 
img_flipped = cv2.flip(fix_img,1)           # horizon 
img_flipped = cv2.flip(fix_img,-1)          # vertical + horizon



""" cv2.imwrite() """
cv2.imwrite('newly_saved.jpg', fix_img)





# fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(111)
# ax.imshow(fix_img)

