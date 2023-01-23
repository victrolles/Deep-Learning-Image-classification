import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential

TRAINING = False

class Classifier:
    def __init__(self):
        # assign save location
        self.checkpoint_path = "model_save/FCL/cp.ckpt"
        self.cp_callback = None
        
        # image size
        self.img_height = 256
        self.img_width = 384
        self.batch_size = 32

        # Datasets
        self.ds_train = None
        self.ds_validation = None
        self.class_names = None

        # Normalization layer
        self.normalization_layer = tf.keras.layers.Rescaling(1./255)

        # Create model
        self.model = None

    def create_dataset(self):
        self.ds_train = tf.keras.preprocessing.image_dataset_from_directory(
            'datasets/weather/',
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=self.batch_size,
            image_size=(self.img_height, self.img_width),
            shuffle=True,
            seed=123,
            validation_split=0.2,
            subset='training',
        )

        self.ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
            'datasets/weather/',
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=self.batch_size,
            image_size=(self.img_height, self.img_width),
            shuffle=True,
            seed=123,
            validation_split=0.2,
            subset='validation',
        )

        self.class_names = self.ds_train.class_names

    def visualize_data(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self.ds_train.take(1):
            for i in range(32):
                ax = plt.subplot(6, 6, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.axis("off")

        plt.show()

    def visualize_img(self, img):
        plt.figure(figsize=(5, 7))
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    def normalize_data(self):
        self.ds_train = self.ds_train.map(lambda x, y: (self.normalization_layer(x), y))
        self.ds_validation = self.ds_validation.map(lambda x, y: (self.normalization_layer(x), y))

    def create_model(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(self.img_height, self.img_width, 3)))
        self.model.add(Dense(units = 512, activation = 'relu'))
        self.model.add(Dense(units = 512, activation = 'relu'))
        self.model.add(Dense(units = len(self.class_names), activation = 'softmax'))

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

    def save_model(self):
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,save_weights_only=True,verbose=1, save_best_only=True)

    def train_model(self):
        self.model.fit(self.ds_train, epochs=15, validation_data=self.ds_validation, callbacks=[self.cp_callback])

    def load_model(self):
        self.model.load_weights(self.checkpoint_path)

    def test_model_on_images(self):
        img = tf.keras.utils.load_img('datasets/test/weather/cloud.png', target_size=(self.img_height, self.img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        img_array = self.normalization_layer(img_array)
        predictions = self.model.predict(img_array)
        # self.visualize_img(img_array[0])
        for idx, elem in enumerate(predictions[0]):
            print("{} : {:.3f}".format(self.class_names[idx], elem*100))

if __name__ == "__main__":
    classifier = Classifier()
    classifier.create_dataset()
    classifier.normalize_data()
    classifier.create_model()
    
    if TRAINING:
        classifier.compile_model()
        classifier.save_model()
        classifier.train_model()
    else:
        classifier.load_model()
        classifier.test_model_on_images()