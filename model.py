import csv
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Conv2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


def load_data(img_path):
    lines_df = pd.read_csv(img_path + 'driving_log.csv')
    lines_df.columns=['center_img','left_img','right_img','steer_angle', 'throttle', 'brake', 'speed']

    X = lines_df[[ 'center_img', 'left_img', 'right_img']].values
    y = lines_df['steer_angle'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=15)

    return X_train, X_valid, y_train, y_valid



def batch_generator(X_paths,y_val,batch_size):
    assert batch_size%6==0, "batch_size must be divisable by six"
    correction = 0.2
    num_lines=len(X_paths)
    images       = np.empty((batch_size,160,320,3))
    measurements = np.empty(batch_size)
    while 1:
        i = 0
        for ix in np.random.permutation(X_paths.shape[0]):
        #for ix in range(X_paths.shape[0]):
          #center camera-----------------------------------------
            center_path = X_paths[ix][0]
            image = cv2.imread(center_path)
            images[i]=image
            center_steering = np.copy(y_val[ix])
            measurements[i]=center_steering
            i=i+1

          #center camera flipped
            images[i]=cv2.flip(image,1)
            measurements[i]=center_steering*-1
            #measurements[i]=5
            i=i+1

          #left camera--------------------------------------------
            left_path  = X_paths[ix][1]
            image = cv2.imread(left_path)
            images[i]=image
            left_steering = center_steering+correction
            measurements[i]=left_steering
            i=i+1

          #left camera flipped
            images[i]=cv2.flip(image,1)
            measurements[i]=left_steering*-1
            i=i+1

          #right camera--------------------------------------------
            right_path = X_paths[ix][2]
            image = cv2.imread(right_path)
            images[i]=image
            right_steering = center_steering-correction
            measurements[i]=right_steering
            i=i+1

          #right camera flipped
            images[i]=cv2.flip(image,1)
            measurements[i]=right_steering*-1
            
            i=i+1
            if i==batch_size:
                i=0
                yield np.array(images), np.array(measurements)
        #yield (images,measurements)
        #yield np.array(images), np.array(measurements)
        #i = 0

def LeNet_Model():
    model =Sequential()
    model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(6,kernel_size=(5, 5),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16,kernel_size=(5, 5),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16,kernel_size=(5, 5),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(120))
    model.add(Dropout(0.2))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def NVIDIA_Model():
    model =Sequential()
    model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    
    model.add(Conv2D(24,kernel_size=(5, 5),strides=(2,2),activation='relu'))
    model.add(Conv2D(36,kernel_size=(5, 5),strides=(2,2),activation='relu'))
    model.add(Conv2D(48,kernel_size=(5, 5),strides=(2,2),activation='relu'))
    model.add(Conv2D(64,kernel_size=(3, 3),activation='relu'))
    model.add(Conv2D(64,kernel_size=(3, 3),activation='relu'))
    
    model.add(Flatten())
    #model.add(Dropout(0.2))
    model.add(Dense(100))
    #model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

model=NVIDIA_Model()
#model=LeNet_Model()
model.compile(loss='mse',optimizer='adam')
img_path='/media/radovan/Samsung_T3/logs/udacity/'
X_train, X_valid, y_train, y_valid=load_data(img_path)
print('There is ',X_train.shape[0],' training images')
batch_size=120
train_gen = batch_generator(X_train,y_train,batch_size)
valid_gen = batch_generator(X_valid,y_valid,batch_size)

history_object = model.fit_generator(train_gen,
                                     steps_per_epoch=len(X_train)/batch_size,
                                     epochs=5, 
                                     verbose=1, 
                                     validation_data=valid_gen, 
                                     validation_steps=len(X_valid)/batch_size,
                                     shuffle=True)
model.save('model.h5')