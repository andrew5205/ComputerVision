
import matplotlib.pyplot as plt 
import cv2 


img = cv2.imread('00-puppy.jpg')

# # cv2 imread() in BGR
plt.imshow(img)
# plt.show()


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
# plt.show()



img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
plt.imshow(img)
# plt.show()


img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()












