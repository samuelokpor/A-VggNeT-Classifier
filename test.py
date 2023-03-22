from keras.metrics import Precision, Recall, CategoricalAccuracy
from dataset import test
from train import model
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf


model.load_weights('model.h5')

pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy: {acc.result().numpy()}')

img = cv2.imread('dogtest.jpg')
resize = cv2.resize(img, (224, 224))
resize = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
resize = resize / 255.0

yhat = model.predict(np.expand_dims(resize, 0))
pred_label = np.argmax(yhat)

if pred_label == 0:
    animal = "cat"
elif pred_label == 1:
    animal = "dog"
elif pred_label == 2:
    animal = "hedgehog"

cv2.putText(img, animal, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Test Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()