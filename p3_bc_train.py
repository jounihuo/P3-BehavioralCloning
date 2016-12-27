import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

#Reading the recorded data locations
file_data = pd.read_csv('driving_log.csv', header=None)
file_data.columns = ['CenterImage','LeftImage','RightImage','SteeringAngle','Throttle','Break','Speed']

width = 320
height = 70

print('Input image size is: ',width,height)

image_paths = file_data['CenterImage'].values
Y = file_data['SteeringAngle'].values
n_images = len(file_data['CenterImage'])

ns = np.arange(n_images)
np.random.shuffle(ns)

Y = Y[ns]
image_paths = image_paths[ns]

print('There is ',len(Y),' images to train.')

def batch_generator(Y, image_paths, batch_size=128):  
    num_rows = n_images
    X_train = np.zeros((batch_size, height, width, 3))
    y_train = np.zeros(batch_size)
    ctr = None
    while 1:
        for j in range(batch_size):
            if ctr is None or ctr >= n_images:
                ctr = 0 # Initialize counter or reset counter if over bounds
            img = Image.open(image_paths[ctr])
            #img = cv2.imread(image_paths[ctr])
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img = img.crop((0, 60, 320, 130))
            img = np.asarray(img.resize((width, height)))
            X_train[j] = img/127.5 - 1.
            y_train[j] = Y[ctr]
            if np.random.choice(2,1)==0:
                X_train[j] = X_train[j][:,:,::-1]
                y_train[j] = -1.0 * y_train[j]                
            ctr += 1
        yield X_train, y_train

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, Flatten, Reshape, ELU
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(16, 3, 3, input_shape=(height, width, 3)))
model.add(ELU())
model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(50, 5, 5, border_mode="same"))
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(50, 5, 5, border_mode="same"))
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(50, 5, 5, border_mode="same"))
model.add(ELU())
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(250))
model.add(Dropout(.25))
#model.add(ELU())
#model.add(Dense(512))
#model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))
model.summary()
# TODO: Compile and train the model here.

adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

history = model.fit_generator(batch_generator(Y, image_paths), samples_per_epoch=n_images, nb_epoch=7)

import json
from keras.models import model_from_json

data = model.to_json()
with open('model.json', 'w') as outfile:
    json.dump(data, outfile)

model.save_weights('model.h5')
print('Model saved!')










