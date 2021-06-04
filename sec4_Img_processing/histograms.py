
import numpy as np 
import matplotlib.pyplot as plt 
import cv2


def display_img(img):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


img_dark_horse = cv2.imread('horse.jpg')            # original BGR openCV
img_show_horse = cv2.cvtColor(img_dark_horse, cv2.COLOR_BGR2RGB)    # convert to RGB for show 


img_rainbow = cv2.imread('rainbow.jpg')
img_show_rainbow = cv2.cvtColor(img_rainbow, cv2.COLOR_BGR2RGB)


img_blue_bricks = cv2.imread('bricks.jpg')
img_show_bricks = cv2.cvtColor(img_blue_bricks, cv2.COLOR_BGR2RGB)



# plt.imshow(img_dark_horse)
# print(img_dark_horse.shape)         # (1800, 2700, 3)
# # plt.show()

# plt.imshow(img_show_horse)
# print(img_show_horse.shape)         # (1800, 2700, 3)
# # plt.show()




# plt.imshow(img_rainbow)
# print(img_rainbow.shape)                # (550, 413, 3)
# # plt.show()

# plt.imshow(img_show_rainbow)
# print(img_show_rainbow.shape)           # (550, 413, 3)
# # plt.show()



# plt.imshow(img_blue_bricks)
# print(img_blue_bricks.shape)         # (950, 1267, 3)
# plt.show()

# plt.imshow(img_show_bricks)
# print(img_show_bricks.shape)         # (950, 1267, 3)
# # plt.show()


print('################################')
##############################################################################################

""" cv2.calcHist([src], channels=[], mask=None, histSize=[256], ranges=[0, 256])"""

# OpenCV BGR
his_val = cv2.calcHist([img_blue_bricks], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
print(his_val.shape)            # (256, 1)
plt.plot(his_val)
plt.show()
##############################################################################################


img = img_blue_bricks
color = ('b', 'g', 'r')

for i, col in enumerate(color):
    histr = cv2.calcHist([img], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0,256])
    # plt.ylim([0,50000])

plt.title('Histogram For Blue_bricks')
plt.show()




img2 = img_dark_horse
color = ('b', 'g', 'r')

for i, col in enumerate(color):
    histr = cv2.calcHist([img2], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
    plt.plot(histr, color=col)
    
    # set limit for each axis
    plt.xlim([0,256])
    plt.ylim([0,50000])

plt.title('Histogram For dark_horse')
plt.show()
##########################################################################################################



img_rainbow = cv2.imread('rainbow.jpg')
img_show_rainbow = cv2.cvtColor(img_rainbow, cv2.COLOR_BGR2RGB)

# print(img_rainbow.shape)            # (550, 413, 3)


""" creat a mask - pass in the shape from img_rainbow """
mask = np.zeros(img_rainbow.shape[:2], np.uint8)
# plt.imshow(mask, cmap='gray')
# plt.show()

mask[300:400, 100:400] = 255
# plt.imshow(mask, cmap='gray')
# plt.show()




img_masked = cv2.bitwise_and(img_rainbow, img_rainbow, mask=mask)
img_show_rainbow_masked = cv2.bitwise_and(img_show_rainbow, img_show_rainbow, mask=mask)


# plt.imshow(img_show_rainbow_masked)
# plt.show()



# opencv BGR
hist_mask_val_red = cv2.calcHist([img_rainbow], channels=[2], mask=mask, histSize=[256], ranges=[0,256])

hist_no_mask_val_red = cv2.calcHist([img_rainbow], channels=[2], mask=None, histSize=[256], ranges=[0,256])

plt.plot(hist_mask_val_red)
plt.title('hist_mask_val_red')
plt.show()


plt.plot(hist_no_mask_val_red)
plt.title('hist_no_mask_val_red')
plt.show()
##########################################################################################################
















