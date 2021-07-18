import numpy as np 
from numpy import genfromtxt

# required
# # pip install keras 
# # pip install tensorflow

data = genfromtxt('bank_note_data.txt', delimiter=',')

# print(data)
# # [[  3.6216    8.6661   -2.8073   -0.44699   0.     ]
# #  [  4.5459    8.1674   -2.4586   -1.4621    0.     ]
# #  [  3.866    -2.6383    1.9242    0.10645   0.     ]
# #  ...
# #  [ -3.7503  -13.4586   17.5932   -2.7771    1.     ]
# #  [ -3.5637   -8.3827   12.393    -1.2823    1.     ]
# #  [ -2.5419   -0.65804   2.6842    1.1952    1.     ]]


""" seperate labels, features"""
# all rows, but start from ele #4 for col
labels = data[:,4]
# print(labels)
# # [0. 0. 0. ... 1. 1. 1.]


features = data[:,0:4]
# print(features)
# # [[  3.6216    8.6661   -2.8073   -0.44699]
# #  [  4.5459    8.1674   -2.4586   -1.4621 ]
# #  [  3.866    -2.6383    1.9242    0.10645]
# #  ...
# #  [ -3.7503  -13.4586   17.5932   -2.7771 ]
# #  [ -3.5637   -8.3827   12.393    -1.2823 ]
# #  [ -2.5419   -0.65804   2.6842    1.1952 ]]



""" general ML paper indicates
cap X: features - 2D array 
lower y: labels - 1D array 
"""
X = features
y = labels


""" split into training set & test set 
by using sklearn.model_selection
"""
from sklearn.model_selection import train_test_split

# train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# print(len(X))               # 1372
# print(len(X_train))         # 919 - 67%
# print(len(X_test))          # 453 - 33%

# print(X_train)
# # [[-0.8734   -0.033118 -0.20165   0.55774 ]
# #  [ 2.0177    1.7982   -2.9581    0.2099  ]
# #  [-0.36038   4.1158    3.1143   -0.37199 ]
# #  ...
# #  [-7.0364    9.2931    0.16594  -4.5396  ]
# #  [-3.4605    2.6901    0.16165  -1.0224  ]
# #  [-3.3582   -7.2404   11.4419   -0.57113 ]]



# print(len(y))               # 1372
# print(len(y_train))         # 919 - 67%
# print(len(y_test))          # 453 - 33%

# print(y_train)
# # [1. 1. 0. 1. 0. 0. 1. 0. 0. 1. 0. 0. 0. 1. 1. 1. 0. 1. 1. 1. 0. 1. 1. 1.
# #  0. 1. 0. 0. 1. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 1. 1. 0. 0. 1. 0. 0. 1. 1.
# #  1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0.
# #  ...
# #  1. 0. 1. 0. 0. 0. 0. 0. 1. 0. 1. 0. 0. 1. 1. 1. 0. 1. 0. 1. 0. 1. 0. 0.
# #  1. 0. 1. 0. 0. 0. 1. 0. 1. 0. 1. 1. 0. 1. 0. 1. 1. 0. 1. 1. 0. 0. 1. 0.
# #  0. 1. 1. 1. 1. 1. 1.]


""" scale the data 
by using klearn.preprocessing 
"""
from sklearn.preprocessing import MinMaxScaler

# use class
scaler_object = MinMaxScaler()

# fit in X_train 
scaler_object.fit(X_train)
# print(scaler_object)            # MinMaxScaler(copy=True, feature=(0,1))

# transform
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

# print(scaled_X_train.max())         # 1.0000000000000002
# print(scaled_X_train.min())         # 0.0




""" adding layers"""
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()

model.add(Dense(4, input_dim=4, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')


"""  start training  """
model.fit(scaled_X_train, y_train, epochs=50, verbose=2)
# print(model.fit(scaled_X_train, y_train, epochs=50, verbose=2))
# # (base) andrews-mbp-4:sec8_deep_learning andrew$ python keras_basics.py 
# # 2021-07-18 13:21:17.952570: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
# # To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# # 2021-07-18 13:21:20.290174: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
# # Epoch 1/50
# # 29/29 - 11s - loss: 0.6895 - accuracy: 0.5517
# # Epoch 2/50
# # 29/29 - 0s - loss: 0.6858 - accuracy: 0.5550
# # Epoch 3/50
# # 29/29 - 0s - loss: 0.6822 - accuracy: 0.5604
# # Epoch 4/50
# # 29/29 - 0s - loss: 0.6786 - accuracy: 0.5702
# # Epoch 5/50
# # 29/29 - 0s - loss: 0.6755 - accuracy: 0.5800
# # Epoch 6/50
# # 29/29 - 0s - loss: 0.6717 - accuracy: 0.6257
# # Epoch 7/50
# # 29/29 - 0s - loss: 0.6679 - accuracy: 0.6376
# # Epoch 8/50
# # 29/29 - 0s - loss: 0.6640 - accuracy: 0.6431
# # Epoch 9/50
# # 29/29 - 0s - loss: 0.6593 - accuracy: 0.6670
# # Epoch 10/50
# # 29/29 - 0s - loss: 0.6545 - accuracy: 0.6714
# # Epoch 11/50
# # 29/29 - 0s - loss: 0.6498 - accuracy: 0.6703
# # Epoch 12/50
# # 29/29 - 0s - loss: 0.6437 - accuracy: 0.6703
# # Epoch 13/50
# # 29/29 - 0s - loss: 0.6378 - accuracy: 0.6659
# # Epoch 14/50
# # 29/29 - 0s - loss: 0.6310 - accuracy: 0.6496
# # Epoch 15/50
# # 29/29 - 0s - loss: 0.6241 - accuracy: 0.6529
# # Epoch 16/50
# # 29/29 - 0s - loss: 0.6164 - accuracy: 0.6594
# # Epoch 17/50
# # 29/29 - 0s - loss: 0.6075 - accuracy: 0.6605
# # Epoch 18/50
# # 29/29 - 0s - loss: 0.5983 - accuracy: 0.6725
# # Epoch 19/50
# # 29/29 - 0s - loss: 0.5884 - accuracy: 0.6844
# # Epoch 20/50
# # 29/29 - 0s - loss: 0.5790 - accuracy: 0.6921
# # Epoch 21/50
# # 29/29 - 0s - loss: 0.5684 - accuracy: 0.6964
# # Epoch 22/50
# # 29/29 - 0s - loss: 0.5584 - accuracy: 0.7051
# # Epoch 23/50
# # 29/29 - 0s - loss: 0.5475 - accuracy: 0.7171
# # Epoch 24/50
# # 29/29 - 0s - loss: 0.5367 - accuracy: 0.7203
# # Epoch 25/50
# # 29/29 - 0s - loss: 0.5259 - accuracy: 0.7432
# # Epoch 26/50
# # 29/29 - 0s - loss: 0.5141 - accuracy: 0.7563
# # Epoch 27/50
# # 29/29 - 0s - loss: 0.5026 - accuracy: 0.7780
# # Epoch 28/50
# # 29/29 - 0s - loss: 0.4903 - accuracy: 0.7813
# # Epoch 29/50
# # 29/29 - 0s - loss: 0.4785 - accuracy: 0.7954
# # Epoch 30/50
# # 29/29 - 0s - loss: 0.4664 - accuracy: 0.8052
# # Epoch 31/50
# # 29/29 - 0s - loss: 0.4535 - accuracy: 0.8118
# # Epoch 32/50
# # 29/29 - 0s - loss: 0.4412 - accuracy: 0.8161
# # Epoch 33/50
# # 29/29 - 0s - loss: 0.4285 - accuracy: 0.8259
# # Epoch 34/50
# # 29/29 - 0s - loss: 0.4159 - accuracy: 0.8324
# # Epoch 35/50
# # 29/29 - 0s - loss: 0.4037 - accuracy: 0.8422
# # Epoch 36/50
# # 29/29 - 0s - loss: 0.3912 - accuracy: 0.8596
# # Epoch 37/50
# # 29/29 - 0s - loss: 0.3781 - accuracy: 0.8553
# # Epoch 38/50
# # 29/29 - 0s - loss: 0.3671 - accuracy: 0.8716
# # Epoch 39/50
# # 29/29 - 0s - loss: 0.3542 - accuracy: 0.8738
# # Epoch 40/50
# # 29/29 - 0s - loss: 0.3412 - accuracy: 0.8868
# # Epoch 41/50
# # 29/29 - 0s - loss: 0.3293 - accuracy: 0.8879
# # Epoch 42/50
# # 29/29 - 0s - loss: 0.3182 - accuracy: 0.8977
# # Epoch 43/50
# # 29/29 - 0s - loss: 0.3064 - accuracy: 0.8999
# # Epoch 44/50
# # 29/29 - 0s - loss: 0.2953 - accuracy: 0.9064
# # Epoch 45/50
# # 29/29 - 0s - loss: 0.2847 - accuracy: 0.9119
# # Epoch 46/50
# # 29/29 - 0s - loss: 0.2748 - accuracy: 0.9119
# # Epoch 47/50
# # 29/29 - 0s - loss: 0.2640 - accuracy: 0.9173
# # Epoch 48/50
# # 29/29 - 0s - loss: 0.2537 - accuracy: 0.9271
# # Epoch 49/50
# # 29/29 - 0s - loss: 0.2443 - accuracy: 0.9314
# # Epoch 50/50
# # 29/29 - 0s - loss: 0.2351 - accuracy: 0.9314
# # <keras.callbacks.History object at 0x7fe6a9653af0>





# print(model.metrics_names)          # ['loss', 'accuracy']



""" making predictions 
by using model.predict_classes(scaled_X_test)
"""
from sklearn.metrics import confusion_matrix,classification_report
predictions = model.predict_classes(scaled_X_test)


confusion_matrix(y_test, predictions)
# print(confusion_matrix(y_test, predictions))
# # [[249   8]
# #  [ 16 180]]


print(classification_report(y_test, predictions))
#               precision    recall  f1-score   support

#          0.0       0.96      0.98      0.97       257
#          1.0       0.97      0.95      0.96       196

#     accuracy                           0.96       453
#    macro avg       0.97      0.96      0.96       453
# weighted avg       0.96      0.96      0.96       453

model.save('mysupermodel.h5')



from keras.models import load_model
newModel = load_model('mysupermodel.h5')





# /opt/anaconda3/lib/python3.8/site-packages/keras/engine/sequential.py:450: UserWarning: `model.predict_classes()` is deprecated and will be removed after 2021-01-01. Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).
#   warnings.warn('`model.predict_classes()` is deprecated and '


