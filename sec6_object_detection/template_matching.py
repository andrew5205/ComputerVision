
# scan larger image for provided template by sliding the template target image accross the larger image


import numpy as np 
import matplotlib.pyplot as plt 
import cv2



full = cv2.imread('sammy.jpg')
full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
# print(type(full))           # <class 'numpy.ndarray'>
# plt.imshow(full)
# plt.show()


# need to be exact part of the larger image
face = cv2.imread('sammy_face.jpg')
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
# print(type(face))           # <class 'numpy.ndarray'>
# plt.imshow(face)
# plt.show()




""" eval() """ 
# eval()


""" 
All 6 methods for comparison in a list 
use eval() function to convert STRING to Function 
methods = [ 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
""" 
methods = [ 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


for m in methods:
    
    """ create a copy """
    full_copy = full.copy()


    method = eval(m)
    
    
    """ template matching """
    # cv2.matchTemplate(large_image, template, method, result)
    res = cv2.matchTemplate(full_copy, face, method)

    """ get loc after template matching """
    # tuple
    # cv2.minMaxLoc(SRC, mask=) -> minVal, maxVal, minLoc, maxLoc
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    """ each method use diff staring point """
    # Note: check doc how those method works with coordinate 
    # this method from eval(m)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc      # (x,y)
    else:
        top_left = max_loc
        
    
    height, width, channels = face.shape
    # print(face.shape)           # (375, 486, 3)
    
    # y+w, x+h
    bottom_right = (top_left[0]+width, top_left[1]+height)


    """ draw tectangle """
    cv2.rectangle(full_copy, top_left, bottom_right, color=(255,0,0), thickness=10)



    """ plot and show images """
    plt.subplot(121)
    plt.imshow(res)
    plt.title('Heatmap of template matching')
    
    plt.subplot(122)
    plt.imshow(full_copy)
    plt.title('Detection of template')
    
    plt.suptitle(m)
    plt.show()

    print('\n')
    print('\n')






# #############################################################################
# full_copy = full.copy()
# test_method = eval('cv2.TM_CCOEFF')
# test_res = cv2.matchTemplate(full_copy, face, test_method)
# plt.imshow(test_res)
# plt.show()
# #bright site indicate exact location where a match is (top left)
# #############################################################################


