import os
import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from keras_custom.models.language_model import lang_model
from keras_custom.generators.generator_wrappers import data_generator
from TRAIN.utils.data_utils import load_classes, data_directory

"""
Source module for training language model
"""

def train_n_val_data_gen(subset, bert_random):
    # data generators
    directory = data_directory(part='train')  # default is train, use val only for debug
    
    if not bert_random:
        wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')
        print('Using regular BERT...\n')
    else:
        wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k_random98.npy')
        print('Using random BERT 98...\n')
        
    gen, steps = data_generator(
                        directory=directory,
                        classes=None,
                        batch_size=128,
                        seed=42,
                        shuffle=True,
                        subset=subset,
                        validation_split=0.1,
                        class_mode='categorical',  # for BERT indexing
                        target_size=(224, 224),
                        preprocessing_function=preprocess_input,
                        horizontal_flip=True, 
                        wordvec_mtx=wordvec_mtx, 
                        simclr_range=False,
                        simclr_augment=False)
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


def execute():
    ###################################################
    lr = 3e-5  # default 3e-5
    lossWs = [0, 0.1, 1, 2, 3, 5, 7, 10]
    bert_random = False
    for lossW in lossWs:
        version = '11-11-20'
        if bert_random is True:
            version = f'{version}-random'
        discrete_frozen = False
        w2_depth = 2
        run_name = f'{version}-lr={str(lr)}-lossW={lossW}'
        print('run_name = ', run_name)
        ###################################################
        # model
        model = lang_model(w2_depth=w2_depth, discrete_frozen=discrete_frozen)
        model.summary()
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt=Adam(lr=lr))
        model.compile(opt,
                    loss=['mse', 'categorical_crossentropy'],
                    loss_weights=[1, lossW],
                    metrics=['acc'])

        # load in trained discrete weights for cases other than 1:1
        if discrete_frozen:
            with open(f'_trained_weights/discrete_weights-{version}-lr={lr}-lossW=1.pkl', 'rb') as f:
                discrete_weights = pickle.load(f)
            model.get_layer('discrete').set_weights([discrete_weights[0], discrete_weights[1]])
            print('loaded trained discrete weights')
        
        # data
        train_gen, train_steps = train_n_val_data_gen(subset='training', bert_random=bert_random)
        val_gen, val_steps = train_n_val_data_gen(subset='validation', bert_random=bert_random)

        # callbacks and fitting
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
    

        ### save weights including w2 dense, semantic, and discrete --- 
        # w2 dense
        for i in range(w2_depth):
            dense_ws = model.get_layer(f'w2_dense_{i}').get_weights()
            with open(f'_trained_weights/w2_dense_{i}-{run_name}.pkl', 'wb') as f:
                pickle.dump(dense_ws, f)

        # semantic
        semantic_ws = model.get_layer('semantic').get_weights()
        with open(f'_trained_weights/semantic_weights-{run_name}.pkl', 'wb') as f:
            pickle.dump(semantic_ws, f)

        # save discrete too if w3 were notfrozen
        if not discrete_frozen:
            discrete_weights = model.get_layer('discrete').get_weights()
            with open(f'_trained_weights/discrete_weights-{run_name}.pkl', 'wb') as f:
                pickle.dump(discrete_weights, f)