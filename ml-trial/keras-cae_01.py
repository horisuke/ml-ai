import os
import glob
import math
import random

import numpy as np
#import matplotlib as plt
import matplotlib.pyplot as plt

from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Input, MaxPool2D, UpSampling2D, Lambda
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator

from tensorflow.python.keras.datasets import mnist
from IPython.display import display_png


def make_masking_noise_data(data_x, persent=0.1):
    size = data_x.shape
    masking = np.random.binomial(n=1, p=persent, size=size)
    return data_x * masking

def make_gaussian_noise_data(data_x, scale=0.8):
    gaussian_data_x = data_x + np.random.normal(loc=0, scale=scale, size=data_x.shape)
    gaussian_data_x = np.clip(gaussian_data_x, 0, 1)
    return gaussian_data_x

# Download the dataset of MNIST
(x_train, _), (x_test, _) = mnist.load_data()

# Check the shapes of MNIST data to be downloaded.
print('x_train.shape: ', x_train.shape)  # (60000, 28, 28)
print('x_test.shape: ', x_test.shape)    # (10000, 28, 28)

# Preprocessing - exchange the scale of data
x_train = x_train.reshape(-1, 28, 28, 1)
x_train = x_train/255
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = x_test/255

# Make data with masking noise
x_train_masked = make_masking_noise_data(x_train)
x_test_masked = make_masking_noise_data(x_test)

# Make data with gaussian noise
x_train_gauss = make_gaussian_noise_data(x_train)
x_test_gauss = make_gaussian_noise_data(x_test)

# Display orginal data, data with masking noise and data with gaussian noise
# display_png(array_to_img(x_train[0]))
# display_png(array_to_img(x_train_gauss[0]))
# display_png(array_to_img(x_train_masked[0]))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title("M_%d" %i)
    plt.axis("off")
    plt.imshow(x_train[i].reshape(28, 28), cmap=None)
plt.show()
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title("M_%d" %i)
    plt.axis("off")
    plt.imshow(x_train_gauss[i].reshape(28, 28), cmap=None)
plt.show()
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title("M_%d" %i)
    plt.axis("off")
    plt.imshow(x_train_masked[i].reshape(28, 28), cmap=None)
plt.show()

# Create the neural network
autoencoder = Sequential()

## Create the encoder parts
autoencoder.add(Conv2D(16, (3,3), 1, activation='relu', padding='same', input_shape=(28, 28, 1)))
autoencoder.add(MaxPool2D((2,2), padding='same'))
autoencoder.add(Conv2D(8, (3,3), 1, activation='relu', padding='same'))
autoencoder.add(MaxPool2D((2,2), padding='same'))

## Create the decoder parts
autoencoder.add(Conv2D(8, (3,3), 1, activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2,2)))
autoencoder.add(Conv2D(16, (3,3), 1, activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2,2)))
autoencoder.add(Conv2D(1, (3,3), 1, activation='sigmoid', padding='same'))

# Setting for learning
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
initial_weights = autoencoder.get_weights()

# Check the summary of network
autoencoder.summary()

# Learn1(Input data:data with gaussian noise, Label data: original data)
autoencoder.fit(x_train_gauss, x_train, epochs=10, batch_size=20, shuffle=True)

# Predict using the network after Learn1
gauss_preds = autoencoder.predict(x_test_gauss)

#  Learn2(Input data:data with masking noise, Label data: original data)
autoencoder.set_weights(initial_weights)
autoencoder.fit(x_train_masked, x_train, epochs=10, batch_size=20, shuffle=True)

# Predict using the network after Learn2
masked_preds = autoencoder.predict(x_test_masked)

# Display the result of learning.
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title("M_%d" %i)
    plt.axis("off")
    plt.imshow(x_test[i].reshape(28, 28), cmap=None)
plt.show()
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title("M_%d" %i)
    plt.axis("off")
    plt.imshow(x_test_gauss[i].reshape(28, 28), cmap=None)
plt.show()
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title("M_%d" %i)
    plt.axis("off")
    plt.imshow(gauss_preds[i].reshape(28, 28), cmap=None)
plt.show()
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title("M_%d" %i)
    plt.axis("off")
    plt.imshow(x_test_masked[i].reshape(28, 28), cmap=None)
plt.show()
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title("M_%d" %i)
    plt.axis("off")
    plt.imshow(masked_preds[i].reshape(28, 28), cmap=None)
plt.show()

