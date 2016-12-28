# Project 3 - Behavioral Cloning

Udacity project for Selfdriving car Nanodegree.

## Introduction

The goal of this project was to train a Keras model with simulator input images to control the steering angle of the simulator car.

![alt text](https://github.com/jounihuo/P3-BehavioralCloning/blob/master/center_2016.jpg "Example image")

## Implementation

Keras was used to create a CNN model that uses the center camera images from the simulator. A joystick was used to control the steering angle. Three things were done to preprocess the images:
- Suffle
- Cropping
- Normalization to -1.0 ... 1.0

![alt text](https://github.com/jounihuo/P3-BehavioralCloning/blob/master/cropped_example_2016.jpg "Cropped image")

Example of a cropped image.

After this a generator was used to read the images from the source folder. To ensure a more robust model the input image can be randomly flipped left to right and the steering angle accordingly. This means that the number of training images can be quite large to ensure smooth path. Images were trained on a model described in the figure below. The code can be found from p3_bc_train.py.

![alt text](https://github.com/jounihuo/P3-BehavioralCloning/blob/master/nn.jpg "Model")

The model consist of two-dimensional convolutions (blue), max pooling layers (orange), dropout layers (purple), dense layer (green) and a single linear output (red). In the first three convolution layers the batches are also normalized. The full description of the model and dimensions is shown in the table below. Exponential Linear Units (ELUs) are used after each convolution to introduce nonlinearities. Model is based on the Cifar example that can be found from [here](http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/) .


Layer (type)    |   Output Shape      |    Param #  |   Connected to |                    
| ------------- |:-------------------:| -----------:|    -----------:|
convolution2d_1 (Convolution2D) | (None, 68, 318, 16)  | 448   |      convolution2d_input_1[0][0]  |    
elu_1 (ELU)                     | (None, 68, 318, 16)  | 0     |      convolution2d_1[0][0]        |   
batchnormalization_1 (BatchNorma| (None, 68, 318, 16)  | 32    |      elu_1[0][0]                  |    
convolution2d_2 (Convolution2D) | (None, 17, 80, 32)   | 32800 |      batchnormalization_1[0][0]   |    
elu_2 (ELU)                     | (None, 17, 80, 32)   | 0     |      convolution2d_2[0][0]        |    
batchnormalization_2 (BatchNorma| (None, 17, 80, 32)   | 64    |      elu_2[0][0]                  |    
convolution2d_3 (Convolution2D) | (None, 9, 40, 64)    | 51264 |      batchnormalization_2[0][0]   |    
elu_3 (ELU)                     | (None, 9, 40, 64)    | 0     |      convolution2d_3[0][0]        |    
batchnormalization_3 (BatchNorma| (None, 9, 40, 64)    | 128   |      elu_3[0][0]                  |    
maxpooling2d_1 (MaxPooling2D)   | (None, 4, 20, 64)    | 0     |      batchnormalization_3[0][0]   |    
convolution2d_4 (Convolution2D) | (None, 4, 20, 50)    | 80050 |      maxpooling2d_1[0][0]         |    
elu_4 (ELU)                     | (None, 4, 20, 50)    | 0     |      convolution2d_4[0][0]        |    
maxpooling2d_2 (MaxPooling2D)   | (None, 2, 10, 50)    | 0     |      elu_4[0][0]                  |    
convolution2d_5 (Convolution2D) | (None, 2, 10, 50)    | 62550 |      maxpooling2d_2[0][0]         |    
elu_5 (ELU)                     | (None, 2, 10, 50)    | 0     |      convolution2d_5[0][0]        |    
maxpooling2d_3 (MaxPooling2D)   | (None, 1, 5, 50)     | 0     |      elu_5[0][0]                  |    
convolution2d_6 (Convolution2D) | (None, 1, 5, 50)     | 62550 |      maxpooling2d_3[0][0]         |    
elu_6 (ELU)                     | (None, 1, 5, 50)     | 0     |      convolution2d_6[0][0]        |    
flatten_1 (Flatten)             | (None, 250)          | 0     |      elu_6[0][0]                  |    
dropout_1 (Dropout)             | (None, 250)          | 0     |      flatten_1[0][0]              |    
elu_7 (ELU)                     | (None, 250)          | 0     |      dropout_1[0][0]              |    
dense_1 (Dense)                 | (None, 250)          | 62750 |      elu_7[0][0]                  |    
dropout_2 (Dropout)             | (None, 250)          | 0     |      dense_1[0][0]                |    
elu_8 (ELU)                     | (None, 250)          | 0     |      dropout_2[0][0]              |    
dense_2 (Dense)                 | (None, 1)            | 251   |      elu_8[0][0]                  |    

Total params: 352887


## Training and behavioral cloning

Model was trained with a local computer using a GTX1060 GPU with 24682 images. A single epoch of 20 000 images took roughly a 22 seconds. After each trainig the model was tested on the test track on the simulator. When the model failed, a number of training images were taken from the critical area. It was quite fast turnaround time for the simulation.

## Model testing and improvements

The model was tested on the simulator track which was also used to collect the images. In addition to normal driving, several recovery patterns were recorded in problem areas. About 6-10 recoveries were required to correct behaviour patterns. It took some effort to ensure that the images recorded did not include false data that would contradict optimal behavior. Adam optimiser was used to limit parameter tuning. After some trial and error the max pooling layer were added to the model which improved the performance and stability of the steering. Also final number of epochs was raised from the initial 5 to 7 when the input image number started to be close to 20000. The validation set also started to loose accuracy after 5 epochs, so the final number was set 6. Initially there was only 5000 images, so over 15000 were collected just from the recovery training.

To battle overfitting, two dropout layers were used. And use train/valid split to indicate performance during training. The final result could only be tested with the simulator and a single number of RMSE or similar does not show the full performance of the model.


