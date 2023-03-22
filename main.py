import cv2
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy
import numpy as np
import tensorflow as tf

#set seed
tf.random.set_seed(0)

#Load image
# img = cv2.imread("data/test.jpg")
# img = cv2.resize(img, (224, 224))


#VGG16
model = keras.Sequential()

#Block 1
model.add(Conv2D(64, kernel_size=(3,3), padding="same",activation="softplus", input_shape=(224,224,3)))
model.add(Conv2D(64, kernel_size=(3,3), padding="same",activation="softplus"))
model.add(MaxPooling2D((2,2), strides=(2,2)))

#block 2
model.add(Conv2D(128, kernel_size=(3,3), padding="same", activation="softplus"))
model.add(Conv2D(128, kernel_size=(3,3), padding="same", activation="softplus"))
model.add(MaxPooling2D((2,2), strides=(2,2)))

#block 3
model.add(Conv2D(256, kernel_size=(3,3), padding="same", activation="softplus"))
model.add(Conv2D(256, kernel_size=(3,3), padding="same", activation="softplus"))
model.add(Conv2D(256, kernel_size=(3,3), padding="same", activation="softplus"))
model.add(MaxPooling2D((2,2), strides=(2,2)))

#block 4
model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="softplus"))
model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="softplus"))
model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="softplus"))
model.add(MaxPooling2D((2,2), strides=(2,2)))

#block 5
model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="softplus"))
model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="softplus"))
model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="softplus"))
model.add(MaxPooling2D((2,2), strides=(2,2)))


#HEAD
model.add(Flatten())
model.add(Dense(4096, activation="softplus"))
model.add(Dense(4096, activation="softplus"))
model.add(Dense(3, activation="softmax"))


model.build()
model.summary()

#Result
# result = model.predict([img.get_numpy_array()])