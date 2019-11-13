
import os
import librosa
import librosa.display
import struct
import pandas as pd
import numpy as np
from datetime import datetime 
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 


def readFileProperties(filename):

    waveFile = open(filename, "rb")

    fmt = waveFile.read(36)
    
    numChannels = struct.unpack('<H', fmt[10:12])[0]
    sampleRate = struct.unpack('<I', fmt[12:16])[0]
    bitDepth = struct.unpack('<H', fmt[22:24])[0]

    waveFile.close()

    return (numChannels, sampleRate, bitDepth)


def extractFeatures(filename):
   
    try:
        audio, sampleRate = librosa.load(filename, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sampleRate, n_mfcc=40)
        
    except Exception as e:
        print(e)
        return None
     
    mfccs = mfccs[:, :215]
    return mfccs


audiodata = []

for i in range(106):
    filename = "wav-audio/nfl-%03d.wav" % i
    metadata = readFileProperties(filename)
    features = extractFeatures(filename)
    audiodata.append([0, metadata[0], metadata[1], metadata[2], features])

for i in range(19):
    filename = "wav-audio/ad2-%03d.wav" % i
    metadata = readFileProperties(filename)
    features = extractFeatures(filename)
    audiodata.append([1, metadata[0], metadata[1], metadata[2], features])

dataFrame = pd.DataFrame(audiodata, columns=['adBool', 'numChannels', 'sampleRate', 'bitDepth', 'features'])
print("\nDATAFRAME")
print(dataFrame)
print()

x = np.array(dataFrame.features.tolist())
y = np.array(dataFrame.adBool.tolist())

print("x shapes is " + str(x.shape))
print("y shapes is " + str(y.shape))

le = LabelEncoder()
y = to_categorical(le.fit_transform(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

num_rows = 40
num_columns = 215
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

# Construct model 
model = Sequential()
model.add(Conv2D(16, (2,2), input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

num_labels = y.shape[1]
# filter_size = 2
model.add(Dense(num_labels, activation='softmax'))



# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Display model architecture summary 
model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100 * score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)



num_epochs = 72
num_batch_size = 256

# checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5', verbose=1, save_best_only=True)

start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)

duration = datetime.now() - start
print("Training completed in time: ", duration)



# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])

