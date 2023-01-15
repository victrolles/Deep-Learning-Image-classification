import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

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

#Train the model

model.fit(ds_train, epochs=10, validation_data=ds_validation)


# img = Image.open('datasets/sunrise325.jpg')

# img.resize((384,256), Image.ANTIALIAS).save('datasets/sunrise325_greyscale.png')
# listsize = []
# for image in os.listdir('datasets/weather/'):
#     if image.endswith('.jpg'):
#         img = Image.open('datasets/weather/'+image)
#         listsize.append(img.size)
#         # img.resize((256,256), Image.ANTIALIAS).save('datasets/'+image[:-4]+'_greyscale.png')

# # print the mean size of the images
# listwidth = []
# listheight = []
# for i in listsize:
#     listwidth.append(i[0])
#     listheight.append(i[1])

# print('mean width: ', sum(listwidth)/len(listwidth))
# print('mean height: ', sum(listheight)/len(listheight))

# print(503/332)