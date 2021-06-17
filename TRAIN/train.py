import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras_custom.models.language_model import lang_model_contrastive
from keras_custom.generators.generator_wrappers import data_generator_v2
from TRAIN.utils.data_utils import load_config, specific_callbacks, data_directory
from TRAIN.utils.saving_utils import save_model_weights

"""
Training script.
"""

def train_n_val_data_gen(config, subset, bert_random=False, generator_type='simclr', sup=None):
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
        generator_type: simclr or vgg16 - finegrain or coarsegrain 
        sup: is None if finegrain, otherwise it is one of the defined superordinates.
    """
    # data generators
    directory = data_directory(part='train')  # default is train, use val only for debug
    if not bert_random:
        wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')
        print('[Check] Using regular BERT...\n')
    else:
        wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k_random98.npy')
        print('[Check] Using random BERT 98...\n')
    
    if 'simclr' in generator_type:
        preprocessing_function = None
        simclr_range = True
    
    elif 'vgg' in generator_type:
        preprocessing_function = preprocess_input
        simclr_range = False

    gen, steps = data_generator_v2(
                            directory=directory,
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
                            wordvec_mtx=wordvec_mtx,
                            simclr_range=simclr_range,
                            simclr_augment=False,
                            sup=sup)
    return gen, steps


def execute(config):
    model = lang_model_contrastive(config)
    # lossWs = [1, 2, 3, 5, 7, 10, 0.1, 0]
    lossWs = [3]
    if 'finegrain' in config['config_version']:
        superordinates = [None]
    else:
        superordinates = ['reptile', 'canidae', 'bird', 'amphibian', 'primate']
    if 'finegrain' in config['config_version'] and len(superordinates) > 1:
        print(f'Error')
        exit()

    every_runtime = []
    for sup in superordinates:
        
        for lossW in lossWs:
            
            # for every lossW, we restart the timer.
            start_time = time.time()

            opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt=tf.keras.optimizers.Adam(lr=config['lr']))
            model.compile(opt,
                        loss=['mse', 'categorical_crossentropy'],
                        loss_weights=[1, lossW],
                        metrics=['acc'])
            # model.compile(tf.keras.optimizers.Adam(lr=config['lr']),
            #             loss=['mse', 'categorical_crossentropy'],
            #             loss_weights=[1, lossW],
            #             metrics=['acc'])

            train_gen, train_steps = train_n_val_data_gen(
                        config=config, 
                        subset='training', 
                        generator_type=config['generator_type'],
                        sup=sup)
            val_gen, val_steps = train_n_val_data_gen(
                        config=config,
                        subset='validation', 
                        generator_type=config['generator_type'],
                        sup=sup)

            if sup is not None:
                lossW = f'{lossW}-{sup}'
            earlystopping, tensorboard = specific_callbacks(config=config, lossW=lossW)

            model.fit(train_gen,
                    epochs=config['epochs'], 
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

            # time it
            end_time = time.time()
            # in hrs.
            duration = (end_time - start_time) / 3600.
            every_runtime.append(duration)
            config_version = config['config_version']
            np.save(f'every_runtime_{config_version}.npy', every_runtime)

