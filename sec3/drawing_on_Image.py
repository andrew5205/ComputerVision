
import numpy as np 
import matplotlib.pyplot as plt
import cv2



img_blank = np.zeros(shape=(512,512,3), dtype=np.int16)

# print(img_blank.shape)              # (512, 512, 3)

plt.imshow(img_blank)
# plt.show()

print(img_blank)
# [[0 0 0]
#   [0 0 0]
#   [0 0 0]
#   ...
#   [0 0 0]
#   [0 0 0]
#   [0 0 0]]]



""" cv2.rectangle(src, pt1=(), pt2=(), color=(), thickness=) 
pt1=() - top left 
pt2=() - bottom right
"""
cv2.rectangle(img_blank, pt1=(380,10), pt2=(500,150), color=(0, 255, 0), thickness=10)
plt.imshow(img_blank)
# plt.show()


cv2.rectangle(img_blank, pt1=(200,200), pt2=(300,300), color=(0, 0, 255), thickness=10)
plt.imshow(img_blank)
# plt.show()




""" cv2.circle(src, center=(),radius=(), color=(),thichness=) """
cv2.circle(img=img_blank, center=(100,100), radius=(50), color=(255,0,0), thickness=10)
plt.imshow(img_blank)
# plt.show()


# full filled thickness=-1
cv2.circle(img=img_blank, center=(400,400), radius=(50), color=(255,0,0), thickness=-1)
plt.imshow(img_blank)
# plt.show()




""" cv2.lint(src, pt1=(), pt2=(), color=(), thickness=) """
cv2.line(img_blank, pt1=(0,0), pt2=(512,512), color=(102,255,255), thickness=20)
plt.imshow(img_blank)
plt.show()


















