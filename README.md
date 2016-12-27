# Project 3 - Behavioral Cloning

Udacity project for Selfdriving car Nanodegree.

## Introduction

The goal of this project was to train a Keras model with simulator input images to control the steering angle of the simulator car.

## Implementation

Keras was used to create a CNN model that uses the center camera images from the simulator. Two thing were done to preprocess the images:
- Cropped the image
- Normalization

After this a generator was used to read the images from the source folder. Images were trained on a model described in the figure 1 below.

## Training and behavioral cloning

Model was trained with a local computer using a GTX1100 GPU. A single epoch of 20 000 images took roughly a 22 seconds. After each trainig the model was tested on the test track on the simulator. When the model failed, a number of training images were taken from the critical area.

## Testing the model
