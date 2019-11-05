### Imorting all necessary libraires:###
import os
import numpy as np
import csv
import pickle
import cv2
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout,Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Convolution2D
from sklearn.preprocessing import LabelBinarizer
from math import ceil

# directory containig dataset provided by Udacity:
DATA_DIR =  '/opt/carnd_p3/data/'

### Loading and Preparing the dataset:###
def load_dataset(filename):
    lines = [] 
    with open(filename,'r') as csvfile:
        readCSV = csv.reader(csvfile)
        next(readCSV, None) 
        for line in readCSV:
            lines.append(line)
        return lines
### Building the generator: ###
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            # offset to add/substract from center image measurement for left/right image measurements:
            offset_angle = 0.3
            for batch_sample in batch_samples:
                for i in range(0,3):
                    name = DATA_DIR + 'IMG/' + batch_sample[i].split('/')[-1]
                    center_image = cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    if(i==0):
                            angles.append(center_angle)
                    elif(i==1):
                            angles.append(center_angle+offset_angle)
                    elif(i==2):
                            angles.append(center_angle-offset_angle)
                   # data augmentation by flippong images and measurements:
                    images.append(cv2.flip(center_image,1))
                    if(i==0):
                            angles.append(center_angle*(-1))
                    elif(i==1):
                            angles.append((center_angle+offset_angle)*(-1))
                    elif(i==2):
                            angles.append((center_angle-offset_angle)*(-1))
                    
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
#########################################################################
data_file = '/opt/carnd_p3/data/driving_log.csv'
data_lines = load_dataset(data_file)
#### split data into train and validation dataset:
train_data, validation_data = train_test_split(data_lines,test_size=0.2)
#### compile and train the model using the generator function:
train_gen = generator(train_data, batch_size=32)
validation_gen = generator(validation_data, batch_size=32)
#### Defining the model: ####
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation:
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
# trim image to only see section with road
model.add(Cropping2D(cropping=((70,25),(0,0))))           
#layers:
# 5 convolutional layers:
model.add(Convolution2D(24,(5,5),strides=(2,2),activation = 'relu'))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation = 'relu'))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation = 'relu'))
model.add(Convolution2D(64,(3,3),activation = 'relu'))
model.add(Convolution2D(64,(3,3),activation = 'relu'))
# dropout to avoid overfitting:
model.add(Dropout(0.8))
#flattening:
model.add(Flatten())
#fully connected layers:
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
#here the Adam optimizer is used:
model.compile(loss='mse', optimizer='adam')

batch_size = 32
#training:
model.fit_generator(train_gen,steps_per_epoch=ceil(len(train_data)/batch_size),validation_data=validation_gen,validation_steps=ceil(len(validation_data)/batch_size),epochs=5, verbose=1)

model.save('model.h5')
print('Done! Model Saved!')
model.summary()

        

