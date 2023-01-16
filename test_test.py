import tensorflow as tf
#load image on tf
img = tf.keras.utils.load_img('datasets/test/weather/sunshine.png', target_size=(256, 384))
img.show()