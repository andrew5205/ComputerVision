
import PIL
import numpy as np 

import matplotlib.pyplot as plt 
# %pylab inline        # 


"""Pyhton image lib """
from PIL import Image

pic = Image.open('00-puppy.jpg')
# pic = Image.open('../source/Computer-vision-with-Python/DATA/00-puppy.jpg')
# pic.show()

print(type(pic))                # <class 'PIL.JpegImagePlugin.JpegImageFile'>



""" Numpy array """
PIL.JpegImagePlugin.JpegImageFile

pic_arr = np.asarray(pic)

print(type(pic_arr))            # <class 'numpy.ndarray'>

print(pic_arr.shape)            # (1300, 1950, 3) -> (h, w, channel)


plt.imshow(pic_arr)
plt.title('matplotlib.pyplot.imshow(pic_arr)', fontweight = 'bold')
# plt.show()
# ctrl + z to exit terminal




"""R G B split """
pic_red = pic_arr.copy()
plt.imshow(pic_red)
# plt.show()


print(pic_red.shape)            # (1300, 1950, 3)

# R G B 
plt.imshow(pic_red[:,:,0])
# plt.show()


"""clor scale"""
""" Red channel val """
print(pic_red[:,:,0])       # 0 no red, pure black - 255 full pure red
# [[95 97 98 ... 25 25 25]
#  [95 96 96 ... 25 25 25]
#  [95 94 94 ... 25 25 25]
#  ...
#  [19 20 20 ... 23 24 24]
#  [20 20 19 ... 23 24 24]
#  [20 19 19 ... 23 24 24]]


plt.imshow(pic_red[:,:,0], cmap='gray')
# plt.show()



# Green channel 
pic_red[:,:,1]
# set green channel to 0 => get rif of all Green 
pic_red[:,:,1] = 0


# Blue channel 
pic_red[:,:,2]
# set green channel to 0 => get rif of all Blue 
pic_red[:,:,2] = 0

plt.imshow(pic_red)
# plt.show()


print(pic_red.shape)            # (1300, 1950, 3) shape still the same


""" Show only one channel """
print(pic_red[:,:,1].shape)     # (1300, 1950)

