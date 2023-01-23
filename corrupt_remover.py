import tensorflow as tf
import os

img_dir="datasets/PetImages"

num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join(img_dir, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jpeg = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jpeg:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)