import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential

TRAINING = True

class GAN:
    def __init__(self):
        # assign save location
        self.checkpoint_path = "model_save/training_3/cp.ckpt"
        self.cp_callback = None
        
        # image size
        self.img_height = 256
        self.img_width = 384

        # Parameters
        self.batch_size = 32
        self.noise_dim = 100
        self.epochs = 15

        # Datasets
        self.dataset = None
        self.class_names = None

        # Normalization layer
        self.normalization_layer = tf.keras.layers.Rescaling(1./255)

        # Create model
        self.generator = None
        self.discriminator = None

        # Create optimizer
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def create_dataset(self):
        self.dataset = tf.keras.preprocessing.image_dataset_from_directory(
            'datasets/weather/',
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=self.batch_size,
            image_size=(self.img_height, self.img_width),
            shuffle=True,
            seed=123,
        )
        self.class_names = self.dataset.class_names

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

    def create_generator(self):
        self.generator = tf.keras.Sequential()
        self.generator.add(tf.keras.layers.Dense(7*11*256, use_bias=False, input_shape=(100,)))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())
        self.generator.add(tf.keras.layers.Reshape((7, 11, 256)))
        self.generator.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())
        self.generator.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())
        self.generator.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    def create_discriminator(self):
        self.discriminator = tf.keras.Sequential()
        self.discriminator.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[self.img_height, self.img_width, 3]))
        self.discriminator.add(tf.keras.layers.LeakyReLU())
        self.discriminator.add(tf.keras.layers.Dropout(0.3))
        self.discriminator.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.discriminator.add(tf.keras.layers.LeakyReLU())
        self.discriminator.add(tf.keras.layers.Dropout(0.3))
        self.discriminator.add(tf.keras.layers.Flatten())
        self.discriminator.add(tf.keras.layers.Dense(1))

    def generate_sample_img(self):
        noise = tf.random.normal([1, 100])
        generated_image = self.generator(noise, training=False)
        self.visualize_img(generated_image[0, :, :, 0])

    def discriminator_loss(self, real_output, fake_output):
        real_loss = tf.keras.losses.categorical_crossentropy(tf.ones_like(real_output), real_output)
        fake_loss = tf.keras.losses.categorical_crossentropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return tf.keras.losses.categorical_crossentropy(tf.ones_like(fake_output), fake_output)

    # def compile_model(self):
    #     self.generator.compile(loss='categorical_crossentropy', optimizer='adam')
    #     self.discriminator.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # def save_model(self):
    #     self.cp_callback_generator = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,save_weights_only=True,verbose=1, save_best_only=True)

    def train_model(self):
        for epoch in range(self.epochs):
            for idx, real_images in enumerate(self.dataset):
                print(epoch, idx)
                # Generate fake images
                noise = tf.random.normal([self.batch_size, self.noise_dim])

                # Train models             
                with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
                    generated_images = self.generator(noise, training=True)
                    images = tf.concat([real_images, generated_images], axis=0)
                    y1 = tf.constant([[1.]] * self.batch_size + [[0.]] * self.batch_size)
                    y_pred = self.discriminator(images, training=True)
                    d_loss = self.discriminator_loss(y1, y_pred)
                    g_loss = self.generator_loss(y_pred)
                gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
                gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
                self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
                self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

                if epoch == 14 and idx == 0:
                    self.visualize_img(generated_images[0, :, :, 0])

    # def load_model(self):
    #     self.generator.load_weights(self.checkpoint_path)

    # def test_model_on_images(self):
    #     img = tf.keras.utils.load_img('datasets/test/weather/cloud.png', target_size=(self.img_height, self.img_width))
    #     img_array = tf.keras.utils.img_to_array(img)
    #     img_array = tf.expand_dims(img_array, 0) # Create a batch
    #     img_array = self.normalization_layer(img_array)
    #     predictions = self.model.predict(img_array)
    #     # self.visualize_img(img_array[0])
    #     for idx, elem in enumerate(predictions[0]):
    #         print("{} : {:.3f}".format(self.class_names[idx], elem*100))

if __name__ == "__main__":
    generator = GAN()
    generator.create_dataset()
    generator.create_generator()
    generator.create_discriminator()
    
    if TRAINING:
        generator.train_model()
    else:
        generator.load_model()
        generator.test_model_on_images()