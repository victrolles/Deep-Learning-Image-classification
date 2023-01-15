import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os

from tensorflow import keras

from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential

#load image
img_height = 256
img_width = 384
batch_size = 32

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'datasets/weather/',
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='training',
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'datasets/weather/',
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='validation',
)

# plt.figure(figsize=(10, 10))
class_names = ds_train.class_names
# for images, labels in ds_train.take(1):
#     for i in range(32):
#         ax = plt.subplot(6, 6, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.axis("off")

# plt.show()

#Standardize the data

normalization_layer = tf.keras.layers.Rescaling(1./255)
ds_train = ds_train.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))

#Configure the dataset for performance

# AUTOTUNE = tf.data.AUTOTUNE
# ds_train = ds_train.cache().prefetch(buffer_size=AUTOTUNE)
# ds_validation = ds_validation.cache().prefetch(buffer_size=AUTOTUNE)

# for image_batch, labels_batch in ds_train:
#     print(image_batch.shape)
#     print(labels_batch.shape)
#     break

#Create the model
model = Sequential()
model.add(Flatten(input_shape=(img_height, img_width, 3)))
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dense(units = 4, activation = 'softmax'))

# model.summary()

#Compile the model

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

# Save the model

checkpoint_path = "model_save/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1, save_best_only=True)

#Train the model

model.fit(ds_train, epochs=5, validation_data=ds_validation, callbacks=[cp_callback])

# Test the model

img = tf.keras.utils.load_img('datasets/test/weather/sunrise.png', target_size=(384, 256))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
print(predictions)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
