import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import yaml

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras_custom.models.language_model import lang_model_contrastive
from keras_custom.generators.generator_wrappers import simple_generator
from TRAIN.utils.data_utils import load_classes, data_directory


"""
Compare how the semantic representations 
from VGG16(fc2) and SIMCLRv2 differ.

We use validation set to save compute.
"""

model = lang_model_contrastive(config=None)
model.compile(tf.keras.optimizers.Adam(),
             loss=['mse'])

gen, steps = simple_generator(
            directory=data_directory(part='val'),
            classes=None,
            batch_size=16,
            seed=42,
            shuffle=True,
            subset=None,
            validation_split=0.0,
            class_mode='sparse',
            preprocessing_function=preprocess_input,
            horizontal_flip=False)

# TODO: the preprocessing should be using simclr not vgg16
# I wonder how would this affect the learning of those fc?