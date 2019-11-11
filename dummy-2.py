import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics

from keras.models import Sequential,Model
from keras.layers import Input,Convolution2D,MaxPooling2D
from keras.layers.core import Dense,Dropout,Activation,Flatten,Lambda


dropout = 0.2

model = Sequential()
model.add(Conv2D(3, (1, 1), input_shape=(40, 40 ,1), activation='relu'))
model.add(Conv2D(12, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(24, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(48, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(dropout))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(dropout))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(dropout))
model.add(Dense(1))

print(model.summary())