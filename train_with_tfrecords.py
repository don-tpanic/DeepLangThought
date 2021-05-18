import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras_custom.models.language_model import lang_model_contrastive
from keras_custom.generators.generator_wrappers import data_generator, data_generator_v2
from TRAIN.utils.data_utils import load_config, specific_callbacks, data_directory
from TRAIN.utils.saving_utils import save_model_weights


lossW = 1
config = load_config(config_version='simclr_finegrain_v2.2.run1')
model = lang_model_contrastive(config)
model.build(input_shape=(1, 2048))
model.compile(tf.keras.optimizers.Adam(lr=config['lr']),
            loss=['mse', 'categorical_crossentropy'],
            loss_weights=[1, lossW],
            metrics=['acc'])

from tfrecords_imagenet_simclr import prepare_dataset
dataset = prepare_dataset().batch(8)

# model.predict(x) this works
model.fit(dataset, verbose=1)



