
import cv2
import numpy as np 
import matplotlib.pyplot as plt

# import time 



"""          """
""" FUNCTION """
"""          """

# A callback is a function that is passed as an argument to other function.
# call back parm
last_time = None
def draw_circle(event, x, y, flags, params):
    # print("event: ", event)
    if event == cv2.EVENT_LBUTTONDOWN:
    # if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(0,255,0),-1)
        # plt.imshow(img)
        # plt.show()
        print(x,y)
        
    # # event val
    # moving -> 0
    # click & hold -> 1
    # release -> 4


cv2.namedWindow(winname='draw_with_mouse')

# cv2.setMouseCallback(winname, function_to_call_back)
cv2.setMouseCallback('draw_with_mouse', draw_circle)




"""                           """
""" Showing Image With OpenCV """ 
"""                           """
""" Script to cv2 Open img in windows properly """
img = np.zeros((512,512,3), np.int8)

while True:
    cv2.imshow('draw_with_mouse', img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
##############################################################################
##############################################################################






