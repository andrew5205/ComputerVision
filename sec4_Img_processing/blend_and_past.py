import matplotlib.pyplot as plt 
import cv2 


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







