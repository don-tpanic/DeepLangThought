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
from TRAIN.utils.data_utils import load_classes, data_directory
from keras_custom.preprocessing.image_dataset import image_dataset_from_directory

"""
Compare how the semantic representations 
from VGG16(fc2) and SIMCLRv2 differ.

We use validation set to save compute.
"""

model = lang_model_contrastive(config=None)
model.compile(tf.keras.optimizers.Adam(),
             loss=['mse'])

train_ds = image_dataset_from_directory(
                data_directory(part='train'),
                validation_split=0.1,
                subset="training",
                seed=42,
                image_size=(224, 224),
                batch_size=1,
                label_mode='categorical_n_bert',
                wordvec_mtx=np.load('data_local/imagenet2vec/imagenet2vec_1k.npy'))

for x, y in train_ds:
    print('it')
    exit()

