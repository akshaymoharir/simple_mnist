
# This file has simplest implementation of ML perceptron using MNIST dataset.
# Model is going to be a single neuron that classifies hand written digits.

import keras
from keras.datasets import mnist
import numpy as np
import cv2
print("keras version:",keras.__version__)

# Load MNIST data
((X_train,Y_train),(X_test,Y_test)) = mnist.load_data()
print("Size of X_train:",X_train.shape)
n_training_images = X_train.shape[0]
print("Number of images in training data:", n_training_images)
random_image_index = np.random.randint(0,n_training_images)
print("Size of random index ", str(random_image_index), " image:",X_train[random_image_index].shape)
random_index_image = cv2.Mat(X_train[random_image_index])
cv2.imshow("Random index image",random_index_image)
cv2.waitKey(0)