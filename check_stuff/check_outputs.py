import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
import time
import numpy as np
import tensorflow as tf
from keras_custom.models.language_model import lang_model_contrastive
from TRAIN.utils.data_utils import load_config, data_directory
from keras_custom.generators import load_tfrecords


def execute(config_version):
    config = load_config(config_version)
    model = lang_model_contrastive(config)
    # prev we didn't have to build, because now
    # we are headless.
    model.build(input_shape=(1, 2048))

    directory = data_directory(part='train', tfrecords=True)
    classes = None

    model.compile(tf.keras.optimizers.Adam(lr=3e-4),
                loss=['mse', 'sparse_categorical_crossentropy'],
                loss_weights=[1, 1],
                metrics=['acc'])

    val_dataset, val_steps = load_tfrecords.prepare_dataset(
                directory=directory,
                classes=classes,
                subset='validation',
                validation_split=0.01,
                batch_size=1,
                sup='canidae')
    
    print(model.predict(val_dataset))


execute(config_version='check_output')




