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

from keras_custom.models.language_model import lang_model_contrastive
from keras_custom.generators.generator_wrappers import data_generator
from TRAIN.utils.data_utils import load_classes, data_directory, load_config


def execute(config):
    lossWs = [0, 0.1, 1, 2, 3, 5, 7, 10]
    # load empty model
    model = lang_model_contrastive(config=config)

    Accuracy_recorder = []
    for lossW in lossWs:

        if lossW == 0:
            Accuracy_recorder.append(0)
            continue

        # load trained model
        model = load_model(config=config, model=model, lossW=lossW)
        # test generator
        gen, steps = test_data_gen(config=config)
        # compute metrics
        top1_acc = compute_acc(model=model, gen=gen)
        Accuracy_recorder.append(top1_acc)

    config_version = config['config_version']
    save_path = f'RESULTS/revision_1/{config_version}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(f'{save_path}/top1_acc.npy', Accuracy_recorder)


def test_data_gen(config):
    """
    Purpose:
    --------
        Return a generator for testing only.
    """
    directory = data_directory(part='val')  # default is train, use val only for debug
    wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')
    gen, steps = data_generator(directory=directory,
                                classes=None,
                                batch_size=config['batch_size'],
                                seed=config['generator_seed'],
                                shuffle=False,
                                subset=None,
                                validation_split=0.0,
                                class_mode='categorical',
                                target_size=(224, 224),
                                preprocessing_function=None,
                                horizontal_flip=False, 
                                wordvec_mtx=wordvec_mtx,
                                simclr_range=True,
                                simclr_augment=False)
    return gen, steps


def load_model(config, model, lossW):
    w2_depth = config['w2_depth'] 
    config_version = config['config_version']

    model.build(input_shape=(1,224,224,3))
    model.summary()

    print(f'**** lossW = {lossW} ****')
    for i in range(w2_depth):
        with open(f'_trained_weights/{config_version}/w2_dense_{i}-{config_version}-lossW={lossW}.pkl', 'rb') as f:
            dense_weights = pickle.load(f)
            model.get_layer(f'w2_dense_{i}').set_weights([dense_weights[0], dense_weights[1]])
            print(f'Loaded layer weights for [w2_dense_{i}]')

    ## semantic weights
    with open(f'_trained_weights/{config_version}/semantic_weights-{config_version}-lossW={lossW}.pkl', 'rb') as f:
        semantic_weights = pickle.load(f)
        model.get_layer('semantic_layer').set_weights([semantic_weights[0], semantic_weights[1]])
        print(f'Loaded layer weights for [semantic]')

    with open(f'_trained_weights/{config_version}/discrete_weights-{config_version}-lossW={lossW}.pkl', 'rb') as f:
        discrete_weights = pickle.load(f)
        model.get_layer('discrete_layer').set_weights([discrete_weights[0], discrete_weights[1]])
        print(f'Loaded layer weights for [discrete]')
    
    model.compile(tf.keras.optimizers.Adam(lr=config['lr']),
                    loss=['mse', 'categorical_crossentropy'],
                    loss_weights=[1, lossW],
                    metrics=['acc'])
    return model


def compute_acc(model, gen):
    _, _, _, _, acc = model.evaluate(gen)
    return acc


if __name__ == '__main__':
    execute()