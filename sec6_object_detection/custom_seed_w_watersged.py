from types import DynamicClassAttribute
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.core.arrayprint import dtype_is_implied


# https://matplotlib.org/examples/color/colormaps_reference.html


# def display(img, cmap='gray'):
#     fig = plt.figure(figsize=(8,8))
#     ax = fig.add_subplot(111)
#     ax.imshow(img, cmap='gray')
#     plt.show()


img_road = cv2.imread('road_image.jpg')

img_road_copy = np.copy(img_road)
# plt.imshow(img_road)
# plt.show()

# print(img_road.shape)           # (600, 800, 3)
# print(img_road.shape[:2])       # (600, 800)

marker_image = np.zeros(img_road.shape[:2], dtype=np.int32)
segments = np.zeros(img_road.shape, dtype=np.uint8)




# Qualitative colormaps
""" from matplotlib import cm """
# print(cm.tab10(0))              # (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0) -> (R, G, B, Alpha)
# print(cm.tab10(0)[:3])          # (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)


def create_rgb(i):
    x = np.array(cm.tab10(0)[:3])*255
    return tuple(x)
# print()

colors = []
for i in range(10):
    colors.append(create_rgb(i))
# print(colors)     # 10 color map
# # [(31.0, 119.0, 180.0), (31.0, 119.0, 180.0), (31.0, 119.0, 180.0), (31.0, 119.0, 180.0), (31.0, 119.0, 180.0), 
# #  (31.0, 119.0, 180.0), (31.0, 119.0, 180.0), (31.0, 119.0, 180.0), (31.0, 119.0, 180.0), (31.0, 119.0, 180.0)]



""" Global Variable """ 
# color choice 
n_markers = 10  # number 0-9
current_markers = 1

#markers update by watershed 
marks_updated = False


""" Callback """ 
def mouse_callback(event, x, y, flags, param):
    global marks_updated
        
    if event == cv2.EVENT_LBUTTONDOWN:
        #markers passed to watershed algo
        cv2.circle(marker_image, (x,y), 10, (current_markers), -1)

        # user sees on the img_road image
        cv2.circle(img_road_copy, (x,y), 10, colors[current_markers], -1)
        
        marks_updated = True
    


""" while True """
cv2.namedWindow('Road Image')
cv2.setMouseCallback('Road Image', mouse_callback)


while True:
    
    cv2.imshow('Watershed Segments', segments)
    cv2.imshow('Road Image', img_road_copy)

    # close windows
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

    # clearing all colors by pressing c key
    elif k == ord('c'):
        img_road_copy = img_road.copy()
        marker_image = np.zeros(img_road.shape[0:2], dtype=np.int32)
        segments = np.zeros(img_road.shape, dtype=np.uint8)
        
    # Update color choice 
    elif k > 0 and chr(k).isdigit():
        current_marker = int(chr(k))
        print(k)

    # Update the markers
    if marks_updated:
        
        marker_img_copy = marker_image.copy()
        cv2.watershed(img_road, marker_img_copy)

        segments = np.zeros(img_road.shape, dtype=np.uint8)

        for color_ind in range(n_markers):
            # coloring segment, np call 
            segments[marker_img_copy == (color_ind)] = colors[color_ind] 
        marks_updated = False



cv2.destroyAllWindows()


