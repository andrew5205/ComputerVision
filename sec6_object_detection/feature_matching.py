

import cv2 
import numpy as np 
import matplotlib.pyplot as plt 



def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()



reeses = cv2.imread('reeses_puffs.png', 0)
# display(reeses)


cereals = cv2.imread('many_cereals.jpg', 0)
# display(cereals)



""" cv2.ORB_create() """
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(reeses, None)
kp2, des2 = orb.detectAndCompute(cereals, None)





bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

single_matches = matches[0]
# print(single_matches)           # <DMatch 0x7ffc9e98ccf0>
# print(single_matches.distance)      # 58.0
# print(len(matches))             # 265

matches = sorted(matches, key=lambda x:x.distance)


""" cv2.drawMatches() """ 
# top # of matched -> matches[0:#]
reeses_matches = cv2.drawMatches(reeses, kp1, cereals, kp2, matches[:25], None, flags=2)


display(reeses_matches)








