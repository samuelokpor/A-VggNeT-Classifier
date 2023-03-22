from main import model, img
from matplotlib import pyplot as plt
import numpy as np

model.summary()

feature_map = model.predict(np.array([img]))

#display Feature Map
for i in range(64):
    feature_img = feature_map[0, :,:,i]
    ax = plt.subplot(8, 8, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(feature_img, cmap="gray")
plt.show()