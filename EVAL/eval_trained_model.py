import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras_custom.models.language_model import lang_model
from keras_custom.generators.generator_wrappers import lang_gen
from EVAL.utils.data_utils import data_directory


"""
Test trained model on white validation set.
"""

def execute():
    # model
    model = lang_model()
    # weights look like [kernel(4096,768), bias(768,)]
    trained_weights = np.load('_trained_weights/semantic_output_weights=29-06-20.npy',
                              allow_pickle=True)
    model.get_layer('semantic_output').set_weights([trained_weights[0], trained_weights[1]])
    model.compile(tf.keras.optimizers.Adam(lr=3e-5),
                  loss=[tf.keras.losses.MSE],
                  metrics='mean_squared_error',
                  )
    
    # test data
    wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')
    directory = data_directory(part='val_white')
    gen, steps = lang_gen(
                        directory=directory,
                        classes=None,
                        batch_size=16,
                        seed=42,
                        shuffle=True,
                        subset=None,
                        validation_split=0,
                        class_mode='categorical',  # only used for lang due to BERT indexing
                        target_size=(224, 224),
                        preprocessing_function=preprocess_input,
                        horizontal_flip=False, 
                        wordvec_mtx=wordvec_mtx)
    
    # eval
    model.evaluate_generator(gen, steps, verbose=1, workers=3)
    

