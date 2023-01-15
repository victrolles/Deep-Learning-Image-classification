import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os

from tensorflow import keras

from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential

# assign location
checkpoint_path = "model_save/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

img_height = 256
img_width = 384
batch_size = 32

img = tf.keras.utils.load_img('datasets/test/weather/sunshine.png', target_size=(384, 256))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

#normalize the data
normalization_layer = tf.keras.layers.Rescaling(1./255)
img_array = normalization_layer(img_array)

# Create model
model = Sequential()
model.add(Flatten(input_shape=(img_height, img_width, 3)))
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dense(units = 4, activation = 'softmax'))

# Load weights
model.load_weights(checkpoint_path)

predictions = model.predict(img_array)
print(predictions)
score = tf.nn.softmax(predictions[0])

class_names = ['cloudy', 'rain', 'shine', 'sunrise']

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
