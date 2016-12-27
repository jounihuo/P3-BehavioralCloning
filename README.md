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

After this a generator was used to read the images from the source folder. To ensure a more robust model the input image can be randomly flipped left to right and the steering angle accordingly. This means that the number of training images can be quite large to ensure smooth path. Images were trained on a model described in the figure 1 below. The code can be found from p3_bc_train.py.

![alt text](https://github.com/jounihuo/P3-BehavioralCloning/blob/master/nn.jpg "Model")

Overall model.

## Training and behavioral cloning

Model was trained with a local computer using a GTX1100 GPU. A single epoch of 20 000 images took roughly a 22 seconds. After each trainig the model was tested on the test track on the simulator. When the model failed, a number of training images were taken from the critical area.

## Model testing and improvements

The model was tested on the simulator track which was also used to collect the images. In addition to normal driving, several recovery patterns were recorded in problem areas. About 6-10 recoveries were required to correct behaviour patterns. It took some effort to ensure that the images recorded did not include false data that would contradict optimal behavior.
