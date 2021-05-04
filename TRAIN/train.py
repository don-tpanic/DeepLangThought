import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import pickle
import yaml

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras_custom.models.language_model import lang_model_contrastive
from keras_custom.generators.generator_wrappers import lang_gen
from TRAIN.utils.data_utils import load_classes, data_directory

"""
TODO: integrate with using VGG16, choice of model needs 
to be just an option not a separate script.

TODO: due to front-end change, input generator needs to work for simclr.
"""

def load_config(config_version):
    with open(os.path.join('configs', f'{config_version}.yaml')) as f:
        config = yaml.safe_load(f)
    print(f'[Check] Loading [{config_version}]')
    return config


def train_n_val_data_gen(subset, bert_random=False):
    # data generators
    directory = data_directory(part='train')  # default is train, use val only for debug
    
    if not bert_random:
        wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')
        print('[Check] Using regular BERT...\n')
    else:
        wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k_random98.npy')
        print('[Check] Using random BERT 98...\n')
        
    gen, steps = lang_gen(
                        directory=directory,
                        classes=None,
                        batch_size=128,
                        seed=42,
                        shuffle=True,
                        subset=subset,
                        validation_split=0.01,
                        class_mode='categorical',  # only used for lang due to BERT indexing
                        target_size=(224, 224),
                        preprocessing_function=preprocess_input,
                        horizontal_flip=True, 
                        wordvec_mtx=wordvec_mtx)
    return gen, steps


def execute():
    config = load_config('test_v1')
    model = lang_model_contrastive(config)
    # model.build((1, 224, 224, 3))
    # model.summary()
    # BUG: no simclr layers visible in model.summary()
    lossW = 1
    model.compile(tf.keras.optimizers.Adam(),
                  loss=['mse', 'categorical_crossentropy'],
                  loss_weights=[1, lossW],
                  metrics=['acc'])

    # data
    train_gen, train_steps = train_n_val_data_gen(subset='validation')
    val_gen, val_steps = train_n_val_data_gen(subset='validation')

    # callbacks and fitting
    # earlystopping, tensorboard = specific_callbacks(run_name=run_name)
    model.fit(train_gen,
                epochs=1, 
                verbose=1, 
                # callbacks=[earlystopping, tensorboard],
                validation_data=val_gen, 
                steps_per_epoch=train_steps,
                validation_steps=val_steps,
                max_queue_size=40, workers=3, 
                use_multiprocessing=False)

