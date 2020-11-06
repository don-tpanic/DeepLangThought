import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras_custom.generators.generator_wrappers import lang_gen
from EVAL.utils.data_utils import data_directory
from EVAL.utils.model_utils import ready_model


"""
Test trained model on white validation set.
"""

def execute():
    ###########################
    part = 'val_white'
    lr = 3e-5
    lossW = [0, 0.1, 1, 2, 3, 5, 7, 10]
    version = '27-7-20'
    #discrete_frozen = False
    w2_depth = 2
    run_name = f'{version}-lr={str(lr)}-lossW={lossW}'
    intersect_layer = 'discrete'  
    # WARNING: semantic is not an option for evaluation
    # because the generate is fixed for [semantic, discrete] outputs.
    ###########################

    # model
    model = ready_model(w2_depth=w2_depth, 
                        run_name=run_name, 
                        lossW=lossW, 
                        intersect_layer=intersect_layer)

    # test data
    wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')
    directory = data_directory(part=part)
    gen, steps = lang_gen(
                        directory=directory,
                        classes=None,
                        batch_size=16,
                        seed=42,
                        shuffle=True,
                        subset=None,
                        validation_split=0,
                        class_mode='categorical',
                        target_size=(224, 224),
                        preprocessing_function=preprocess_input,
                        horizontal_flip=False, 
                        wordvec_mtx=wordvec_mtx)

    model.evaluate_generator(gen, steps, verbose=1, workers=3)



