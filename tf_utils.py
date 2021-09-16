import tensorflow as tf
from PIL import Image
import numpy as np

def convert_to_img_tensor(image: Image.Image) -> tf.Tensor:
    max_dim = 512
    img_tensor = tf.constant(image, dtype=np.uint8)
    img_tensor = tf.image.convert_image_dtype(img_tensor, dtype=tf.float32)

    shape = tf.cast(tf.shape(img_tensor)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img_tensor = tf.image.resize(img_tensor, new_shape)
    img_tensor = img_tensor[tf.newaxis, :]
    return img_tensor


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)
