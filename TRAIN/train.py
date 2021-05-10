import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras_custom.models.language_model import lang_model_contrastive
from keras_custom.generators.generator_wrappers import simclr_gen, lang_gen, sup_gen
from TRAIN.utils.data_utils import load_config, specific_callbacks, data_directory
from TRAIN.utils.saving_utils import save_model_weights


"""
TODO: integrate with using VGG16, choice of model needs 
to be just an option not a separate script.
"""

def train_n_val_data_gen(config, subset, bert_random=False, generator_type='simclr'):
    """
    Purpose:
    --------
        Return a generator can be used for one of the following tasks
            1. VGG front end with either finegrain or coarsegrain labels
            2. simclr front end with either finegrain or coarsegrain labels
    
    inputs:
    -------
        subset: training or validation
        bert_random: True or False
        generator_type: simclr 
        # TODO: this needs regrouped.
    """
    # data generators
    directory = data_directory(part='train')  # default is train, use val only for debug
    if not bert_random:
        wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')
        print('[Check] Using regular BERT...\n')
    else:
        wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k_random98.npy')
        print('[Check] Using random BERT 98...\n')
    
    if generator_type == 'simclr':
        generator = simclr_gen
        preprocessing_function = None
    elif generator_type == 'finegrain':
        generator = lang_gen
        preprocessing_function = preprocess_input
    elif generator_type == 'coarsegrain':
        generator = sup_gen
        preprocessing_function = preprocess_input

    gen, steps = generator(directory=directory,
                           classes=None,
                           batch_size=config['batch_size'],
                           seed=config['generator_seed'],
                           shuffle=True,
                           subset=subset,
                           validation_split=config['validation_split'],
                           class_mode='categorical',
                           target_size=(224, 224),
                           preprocessing_function=preprocessing_function,
                           horizontal_flip=True, 
                           wordvec_mtx=wordvec_mtx)
    return gen, steps


def execute():
    config = load_config('simclr_finegrain_v1.1.run1')
    model = lang_model_contrastive(config)
    # lossWs = [0, 0.1, 1, 2, 3, 5, 7, 10]
    # lossWs = [0, 0.1, 1, 2]
    lossWs = [3, 5, 7, 10]
    for lossW in lossWs:
        model.compile(tf.keras.optimizers.Adam(lr=config['lr']),
                    loss=['mse', 'categorical_crossentropy'],
                    loss_weights=[1, lossW],
                    metrics=['acc'])
        train_gen, train_steps = train_n_val_data_gen(
                    config=config, 
                    subset='training', 
                    generator_type=config['generator_type'])
        val_gen, val_steps = train_n_val_data_gen(
                    config=config,
                    subset='validation', 
                    generator_type=config['generator_type'])
        earlystopping, tensorboard = specific_callbacks(config=config, lossW=lossW)
        model.fit(train_gen,
                  epochs=500, 
                  verbose=1, 
                  callbacks=[earlystopping, tensorboard],
                  validation_data=val_gen,
                  steps_per_epoch=train_steps,
                  validation_steps=val_steps,
                  max_queue_size=40, 
                  workers=3, 
                  use_multiprocessing=False)

        # save trained weights
        save_model_weights(model=model, config=config, lossW=lossW)


