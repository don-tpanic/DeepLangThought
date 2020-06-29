import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from keras_custom.models.language_model import lang_model
from keras_custom.generators.generator_wrappers import lang_gen
from keras_custom.losses import pearson_loss
from TRAIN.utils.data_utils import load_classes, data_directory

"""
Source module for training language model
"""

def train_n_val_data_gen(subset):
    # data generators
    directory = data_directory(part='train')  # default is train, use val only for debug
    wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')
    gen, steps = lang_gen(
                        directory=directory,
                        classes=None,
                        batch_size=16,
                        seed=42,
                        shuffle=True,
                        subset=subset,
                        validation_split=0.1,
                        class_mode='categorical',  # only used for lang due to BERT indexing
                        target_size=(224, 224),
                        preprocessing_function=preprocess_input,
                        horizontal_flip=True, 
                        wordvec_mtx=wordvec_mtx)
    return gen, steps


def specific_callbacks(run_name):
    """
    Define earlystopping and tensorboard.
    """
    earlystopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    min_delta=0, 
                    patience=5, 
                    verbose=2, 
                    mode='min',
                    baseline=None, 
                    restore_best_weights=True
                    )
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f'log/{run_name}')
    return earlystopping, tensorboard


# def execute():

#     train_gen, train_steps = train_n_val_data_gen(subset='training')
#     val_gen, val_steps = train_n_val_data_gen(subset='validation')

#     model = lang_model()

#     model.compile(Adam(),
#                   loss=[tf.keras.losses.MSE, 
#                         tf.keras.losses.CategoricalCrossentropy()],
#                   loss_weights={'semantic_output':1,
#                                 'label_output': 0},
#                   )

#     earlystopping, tensorboard = specific_callbacks()

#     model.fit(train_gen,
#                 epochs=500, 
#                 verbose=1, 
#                 #callbacks=[earlystopping, tensorboard],
#                 validation_data=val_gen, 
#                 steps_per_epoch=train_steps,
#                 validation_steps=val_steps, 
#                 max_queue_size=40, workers=3, 
#                 use_multiprocessing=False)


def execute(run_name='semantic_output_only'):
    train_gen, train_steps = train_n_val_data_gen(subset='training')
    val_gen, val_steps = train_n_val_data_gen(subset='validation')

    model = lang_model()

    model.compile(Adam(lr=3e-5),
                  loss=[tf.keras.losses.MSE],
                  metrics='mean_squared_error',
                  )

    earlystopping, tensorboard = specific_callbacks(run_name=run_name)

    model.fit(train_gen,
                epochs=500, 
                verbose=1, 
                callbacks=[earlystopping, tensorboard],
                validation_data=val_gen, 
                steps_per_epoch=train_steps,
                validation_steps=val_steps, 
                max_queue_size=40, workers=3, 
                use_multiprocessing=False)
    
    semantic_output_ws = model.get_layer('semantic_output').get_weights()
    np.save('trained_weights/semantic_output_weights.npy', semantic_output_ws)
    print('weights saved.')