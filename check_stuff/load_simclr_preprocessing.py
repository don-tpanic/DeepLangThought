import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras_custom.generators.simclr_preprocessing import preprocess_image


def execute():
    img = load_img("check_stuff/test_img.png")
    x = img_to_array(img, data_format="channels_last")
    print(x)
    print(type(x))  
    x = tf.convert_to_tensor(x)
    print(type(x))

    x = preprocess_image(image=x, height=224, width=224, is_training=False, color_distort=False)
    print(np.unique(x.numpy()))
    print(type(x))