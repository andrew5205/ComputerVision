
import numpy as np 
import matplotlib.pyplot as plt
import cv2


# https://docs.opencv.org/master/dc/da5/tutorial_py_drawing_functions.html

# https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9


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



""" cv2.rectangle(src, pt1=(), pt2=(), color=(), thickness=, lineType=, shift=) 
pt1=() - top left 
pt2=() - bottom right
"""
cv2.rectangle(img_blank, pt1=(380,10), pt2=(500,150), color=(0, 255, 0), thickness=10)
plt.imshow(img_blank)
# plt.show()


cv2.rectangle(img_blank, pt1=(200,200), pt2=(300,300), color=(0, 0, 255), thickness=10)
plt.imshow(img_blank)
# plt.show()




""" cv2.circle(src, center=(),radius=(), color=(), thichness=, lineType=, shift=) """
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
# plt.show()





""" Font """ 
font = cv2.FONT_HERSHEY_SIMPLEX

""" cv2.putText(src, text, org, FontFace, FontScale, color, thickness, lineType) """
cv2.putText(img_blank, text='Hello', org=(10,500), fontFace=font, fontScale=4, color=(255,255,255), thickness=5, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
plt.imshow(img_blank)
# plt.show()



"""cv2.polygan() """
# dtype =>  data type, default numpy.float64
# np.int32 -> int 
# np.float64 -> float 
img_new_blank = np.zeros(shape=(512,512,3), dtype=np.int32)
plt.imshow(img_new_blank)


vertices = np.array([[100, 300], [200, 200], [400, 300], [200, 400]], dtype=np.int32)
# plt.show()
print(vertices.shape)               # (4, 2)

""" re-shape into 3 channel """
pts = vertices.reshape((-1, 1, 2))
print(pts.shape)                    # (4, 1, 2)
print(pts)
# [[[100 300]]

#  [[200 200]]

#  [[400 300]]

#  [[200 400]]]


cv2.polylines(img_new_blank, [pts], isClosed=True, color=(255,0,0), thickness=5)
# cv2.polylines(img_new_blank, [pts], isClosed=False, color=(255,0,0), thickness=5)
plt.imshow(img_new_blank)
# plt.show()






