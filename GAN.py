import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential

TRAINING = True

class GAN:
    def __init__(self):
        # assign save location
        self.checkpoint_path_gen = "model_save/GAN/gen/"
        self.checkpoint_path_disc = "model_save/GAN/disc/"
        self.cp_callback = None
        
        # image size
        self.img_height = 128
        self.img_width = 128

        # Parameters
        self.batch_size = 256
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
            'datasets/PetImages/',
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=self.batch_size,
            image_size=(self.img_height, self.img_width),
            shuffle=True,
            seed=123,
        )
        print(len(np.concatenate([i for x, i in self.dataset], axis=0)))
        for idx, (real_images, labels) in enumerate(self.dataset):
            print("idx", idx)
        self.class_names = self.dataset.class_names

    def visualize_data(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self.dataset.take(1):
            for i in range(32):
                ax = plt.subplot(6, 6, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.axis("off")

        plt.show()

    def visualize_data_batch(self, bach_images):
        for i in range(bach_images.shape[0]):
            ax = plt.subplot(6, 6, i + 1)
            plt.imshow(bach_images[i])
            plt.axis("off")

        plt.show()

    def visualize_img(self, img):
        plt.figure(figsize=(5, 7))
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    def cache_data(self):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.dataset = self.dataset.cache().prefetch(buffer_size=AUTOTUNE)

    def create_generator(self):
        self.generator = tf.keras.Sequential()
        self.generator.add(tf.keras.layers.Dense(4*4*256, use_bias=False, input_shape=(self.noise_dim,)))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())
        self.generator.add(tf.keras.layers.Reshape((4, 4, 256)))
        self.generator.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())
        self.generator.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())
        self.generator.add(tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())
        self.generator.add(tf.keras.layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())
        self.generator.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

        # assert self.generator.output_shape == (None, 256, 384, 3)

    def create_discriminator(self):
        self.discriminator = tf.keras.Sequential()
        self.discriminator.add(tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=[self.img_height, self.img_width, 3]))
        self.discriminator.add(tf.keras.layers.LeakyReLU())
        self.discriminator.add(tf.keras.layers.Dropout(0.2))
        self.discriminator.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        self.discriminator.add(tf.keras.layers.LeakyReLU())
        self.discriminator.add(tf.keras.layers.Dropout(0.2))
        self.discriminator.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.discriminator.add(tf.keras.layers.LeakyReLU())
        self.discriminator.add(tf.keras.layers.Dropout(0.2))
        self.discriminator.add(tf.keras.layers.Flatten())
        self.discriminator.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def train_model(self):
        best_loss_gen = np.inf
        best_loss_disc = np.inf
        print("starts training...")
        for epoch in range(self.epochs):
            for idx, (real_images, labels) in enumerate(self.dataset):
                print(epoch, idx)

                mini_batch_size = real_images.shape[0]
                noise = tf.random.normal([mini_batch_size, self.noise_dim])

                # Train models             
                with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:

                    # Generate a batch (32) of new images
                    generated_images = self.generator(noise, training=True)

                    # Visualize the generated images
                    # print("one generated image")
                    # self.visualize_img(generated_images[0])
                    # print("one real image")
                    # self.visualize_img(real_images[0].numpy().astype("uint8"))
                    # print("generated_images")
                    # self.visualize_data_batch(generated_images)
                    # print("real_images")
                    # self.visualize_data_batch(real_images)

                    # Train the discriminator
                    real_output = self.discriminator(real_images, training=True)
                    fake_output = self.discriminator(generated_images, training=True)

                    disc_loss = self.discriminator_loss(real_output, fake_output)
                    gen_loss = self.generator_loss(fake_output)

                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

                self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
                self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

                if disc_loss < best_loss_disc:
                    best_loss_disc = disc_loss
                    self.discriminator.save_weights(self.checkpoint_path_disc)

                if gen_loss < best_loss_gen:
                    best_loss_gen = gen_loss
                    self.generator.save_weights(self.checkpoint_path_gen)


                # if epoch == 1 and idx == 1:
                #     print("one generated image")
                #     self.visualize_img(generated_images[0])
                #     print("one real image")
                #     self.visualize_img(real_images[0].numpy().astype("uint8"))
                # elif epoch == 5 and idx == 1:
                #     print("one generated image")
                #     self.visualize_img(generated_images[0])
                #     print("one real image")
                #     self.visualize_img(real_images[0].numpy().astype("uint8"))
                # elif epoch == 9 and idx == 1:
                #     print("one generated image")
                #     self.visualize_img(generated_images[0])
                #     print("one real image")
                #     self.visualize_img(real_images[0].numpy().astype("uint8"))

    def load_model(self):
        self.discriminator.load_weights(self.checkpoint_path_disc)
        self.generator.load_weights(self.checkpoint_path_gen)

    def test_model_on_images(self):
        noise = tf.random.normal([1, 100])
        generated_image = self.generator(noise, training=False)
        self.visualize_img(generated_image[0])
        self.visualize_data_batch(generated_image)

if __name__ == "__main__":
    generator = GAN()
    generator.create_dataset()
    generator.cache_data()
    generator.create_generator()
    generator.create_discriminator()
    
    if TRAINING:
        generator.train_model()
    else:
        generator.load_model()
        generator.test_model_on_images()