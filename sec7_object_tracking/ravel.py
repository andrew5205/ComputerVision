
import numpy as np
from numpy.core.fromnumeric import reshape 


row1 = [1,2,3,4,5]
row2 = [6,7,8,9,10]
row3 = [11,12,13,14,15]
row4 = [16,17,18,19,20]
row5 = [21,22,23,24,25]

matrix = np.array([row1, row2, row3, row4,row5])
print(matrix)
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 10]
#  [11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]]


# slice 
# array[row, col]
print(matrix[:,2:4:1])          # starting:ending val: step size
# [[ 3  4]
#  [ 8  9]
#  [13 14]
#  [18 19]
#  [23 24]]


# negative index< reverse 
# array[row, col]
print(matrix[:,-2:-4:-1])
# [[ 4  3]
#  [ 9  8]
#  [14 13]
#  [19 18]
#  [24 23]]


# boolean
grater_than_five = matrix >5
print(grater_than_five)
# [[False False False False False]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]]
print(matrix[grater_than_five])             # [ 6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25]









#############################################################################
""" np.ravel() """ 
# A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.


x = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(np.ravel(x))              # [1 2 3 4 5 6 7 8 9]
y = np.ravel(x)
print(y)

# print(x.reshape(1))
print(x.reshape(-1))            # [1 2 3 4 5 6 7 8 9]


print(np.ravel(x, order='C'))       # [1 2 3 4 5 6 7 8 9] row first 
print(np.ravel(x, order='F'))       # [1 4 7 2 5 8 3 6 9] col first 
print(np.ravel(x, order='A'))       # [1 2 3 4 5 6 7 8 9]
print(np.ravel(x, order='K'))       # [1 2 3 4 5 6 7 8 9]


print(x.shape)              # (3, 3)
#############################################################################

""" np.reshape() """
a = np.arange(6).reshape(3,2)
print(a)
# [[0 1]
#  [2 3]
#  [4 5]]

# np.reshape()
print(a.reshape(2,3))
# [[0 1 2]
#  [3 4 5]]


print(a.reshape(-1))            # [0 1 2 3 4 5]




b = np.arange(12).reshape(3,4)
print(np.reshape(b, (6,2)))         # [[ 0  1], [ 2  3], [ 4  5], [ 6  7], [ 8  9], [10 11]]
# print(b.reshape(6,2))


