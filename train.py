
import tensorflow as tf
from dataset import train, val
import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.metrics import Precision, Recall, CategoricalAccuracy
import pandas as pd

# Load the pre-trained VGG16 model
base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False


#add a new classifier head
x = base_model.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(4096, activation="softplus")(x)
x = keras.layers.Dense(4096, activation="softplus")(x)
predictions = keras.layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[CategoricalAccuracy(), Precision(), Recall(), ])
model.summary()

#define callbacks
logdir = 'logs'
csv_logger = CSVLogger('training_history.csv')
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

# #Train the Model
# history = model.fit(train, epochs=20, validation_data=val, callbacks=[csv_logger, checkpoint])

# #save the history in a datframe
# history_df = pd.DataFrame(history.history)
# history_df.to_csv('training_history_df.csv', index=False)