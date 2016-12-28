import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

#Reading the recorded data locations with pandas
file_data = pd.read_csv('driving_log.csv', header=None)
file_data.columns = ['CenterImage','LeftImage','RightImage',
                     'SteeringAngle','Throttle','Break','Speed']

#The final image size to be used
width = 320
height = 70
print('Input image size is: ',width,height)

#Separating the 
image_paths = file_data['CenterImage'].values
Y = file_data['SteeringAngle'].values
n_images = len(file_data['CenterImage'])

#Suffle the images and angles
ns = np.arange(n_images)
np.random.shuffle(ns)
Y = Y[ns]
image_paths = image_paths[ns]
print('There is ',len(Y),' images to train.')

#Creating a validation set
image_paths, image_paths_val, Y, Y_val = train_test_split(image_paths, Y, 
                                                          random_state=123, test_size=0.10)

n_images = len(Y)

#Creating a generator for the training
def batch_generator(Y, image_paths, batch_size=128):  
    num_rows = n_images
    X_train = np.zeros((batch_size, height, width, 3))
    y_train = np.zeros(batch_size)
    ctr = None
    while 1:
        for j in range(batch_size):
            if ctr is None or ctr >= n_images:
                ctr = 0
            img = Image.open(image_paths[ctr])
            #Image is cropped to have only the road visible
            img = img.crop((0, 60, 320, 130))
            img = np.asarray(img.resize((width, height)))
            #Simple normalisation of the RGB image
            X_train[j] = img/127.5 - 1.
            y_train[j] = Y[ctr]
            #Randomly flip the image and steering angle to increase accuracy
            if np.random.choice(2,1)==0:
                X_train[j] = X_train[j][:,:,::-1]
                y_train[j] = -1.0 * y_train[j]                
            ctr += 1
        yield X_train, y_train

def val_batch_generator(Y, image_paths, batch_size=128):  
    num_rows = len(Y)
    X_train = np.zeros((batch_size, height, width, 3))
    y_train = np.zeros(batch_size)
    ctr = None
    while 1:
        for j in range(batch_size):
            if ctr is None or ctr >= n_images:
                ctr = 0
            img = Image.open(image_paths[ctr])
            #Image is cropped to have only the road visible
            img = img.crop((0, 60, 320, 130))
            img = np.asarray(img.resize((width, height)))
            #Simple normalisation of the RGB image
            X_train[j] = img/127.5 - 1.
            y_train[j] = Y[ctr]
            #Randomly flip the image and steering angle to increase accuracy
            if np.random.choice(2,1)==0:
                X_train[j] = X_train[j][:,:,::-1]
                y_train[j] = -1.0 * y_train[j]                
            ctr += 1
        yield (X_train, y_train)

#Importing Keras modules and functions
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, Flatten, Reshape, ELU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

#Creating the model
model = Sequential()
model.add(Conv2D(16, 3, 3, input_shape=(height, width, 3)))
model.add(ELU())
model.add(BatchNormalization())
model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(BatchNormalization())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(BatchNormalization())
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
model.add(ELU())
model.add(Dense(1))
model.summary()

#Training the model with Adam
adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

history = model.fit_generator(batch_generator(Y, image_paths), 
                              samples_per_epoch=n_images, 
                              nb_epoch=6, 
                              validation_data = next(val_batch_generator(Y_val, image_paths_val, len(Y_val))))

#Saving the model and the weights
import json
from keras.models import model_from_json

data = model.to_json()
with open('model.json', 'w') as outfile:
    json.dump(data, outfile)

model.save_weights('model.h5')
print('Model saved!')










