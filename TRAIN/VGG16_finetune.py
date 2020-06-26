import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from keras_custom.generators.generator_wrapers import simple_generator
from TRAIN.utils.data_utils import load_classes, data_directory

"""
Due to the pre-trained weights are suboptimal,
here we free up the entire VGG16 and fine tune it
to its absolute best and save the weights for 
future use.
"""


def train_n_val_data_gen(subset):
    # data generators
    train_directory = data_directory()
    gen, steps = simple_generator(
                        directory=train_directory,
                        classes=None,
                        batch_size=16,
                        seed=42,
                        shuffle=True,
                        subset=subset,
                        validation_split=0.1,
                        class_mode='sparse',
                        target_size=(224, 224),
                        preprocessing_function=preprocess_input,
                        horizontal_flip=True,    
                    )
    return gen, steps


def specific_callbacks():
    """
    Define earlystopping and tensorboard.
    """
    earlystopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    min_delta=0, 
                    patience=10, 
                    verbose=2, 
                    mode='min',
                    baseline=None, 
                    restore_best_weights=True
                    )
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log/VGG16finetune')
    return earlystopping, tensorboard


def execute():
    # model
    model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt=Adam(lr=0.0003, epsilon=1.))
    model.compile(opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['acc', 'sparse_top_k_categorical_accuracy'],
                  )
    model.summary()

    # data generators
    train_gen, train_steps = train_n_val_data_gen(subset='training')
    val_gen, val_steps = train_n_val_data_gen(subset='validation')

    # fit
    earlystopping, tensorboard = specific_callbacks()
    model.fit_generator(train_gen,
                        epochs=500, 
                        verbose=1, 
                        callbacks=[earlystopping, tensorboard],
                        validation_data=val_gen, 
                        steps_per_epoch=train_steps,
                        validation_steps=val_steps, 
                        max_queue_size=40, workers=3, 
                        use_multiprocessing=False)

    # save model weights
    model.save_weights('VGG16_finetuned_fullmodelWeights.h5')