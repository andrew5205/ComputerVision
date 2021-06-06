
import cv2
import numpy as np
import matplotlib.pyplot as plt 

# canny algorithm
# John Canny - multi-stage algorithm


img = cv2.imread('sammy_face.jpg')
# print(type(img))            # <class 'numpy.ndarray'>
# plt.imshow(img)
# plt.show()


""" cv2.canny(src, thr1, thr2) """ 
edges = cv2.Canny(image=img, threshold1=127, threshold2=127)
plt.imshow(edges)
# plt.show()

######################################################################################
# formula to select threshold 

med_val = np.median(img)
print(med_val)          # 64.0

# set lower threshold to either 0 or 70% of the median val whichever is greater 
lower = int(max(0, 0.7*med_val))
# set upper threshold to either 130% of median or the max 255, whichever is smaller  
upper = int(min(255, 1.3*med_val))


egdes = cv2.Canny(image=img, threshold1=lower, threshold2=upper)
# egdes = cv2.Canny(image=img, threshold1=lower, threshold2=upper+100)
plt.imshow(edges)
# plt.show()
######################################################################################

""" blur source image first, and then use formula """ 
img_blurred = cv2.blur(img, ksize=(5,5))
egdes = cv2.Canny(image=img_blurred, threshold1=lower, threshold2=upper+50)
plt.imshow(edges)
# plt.show()












