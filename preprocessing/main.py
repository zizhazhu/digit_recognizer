import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

import sklearn.preprocessing as pre

#setting
LEARNING_RATE = 1e-4
TRAINING_ITERATIONS = 2500

DROPOUT = 0.5
BATCH_SIZE = 50

VALIDATION_SIZE = 2000

IMAGE_TO_DISPLAY = 10

# read data
data = pd.read_csv('./input/train.csv')

print('data({0[0]},{0[1]})'.format(data.shape))
#print(data.head())

images = data.iloc[:,1:].values
images = images.astype(np.float)
# normorlize
images = np.multiply(images, 1.0 / 255.0)
print('image({0[0]},{0[1]})'.format(images.shape))

image_size = images.shape[1]
print('image_size => {0}'.format(image_size))
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
print('image_width => {0}\nimage_height => {1}'.format(image_width, image_height))

def display(img):
    one_image = img.reshape(image_width, image_height)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    plt.show()

#display(images[IMAGE_TO_DISPLAY])

labels_flat = data[[0]].values.ravel()
print('labels_flat({0})'.format(len(labels_flat)))
print('labels_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY, labels_flat[IMAGE_TO_DISPLAY]))

labels_count = np.unique(labels_flat).shape[0]
print('labels_count => {0}'.format(labels_count))

# transform lables_flat
labels = pre.OneHotEncoder().fit_transform(labels_flat.reshape((-1,1))).toarray()
labels = labels.astype(np.uint8)

print('labels({0[0]}, {0[1]})'.format(labels.shape))
print('labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY, labels[IMAGE_TO_DISPLAY]))

validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]
train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]
print('train_images({0[0]}, {0[1]})'.format(train_images.shape))
print('validation_images({0[0]}, {0[1]})'.format(validation_images.shape))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')