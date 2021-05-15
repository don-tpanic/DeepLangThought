import os
import numpy as np
import matplotlib.pyplot as plt
from seaborn import violinplot
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras_custom.generators.generator_wrappers import data_generator
from EVAL.utils.data_utils import data_directory, load_classes
from EVAL.utils.model_utils import ready_model

"""
Compute class-level semantic activations out of trained models (vgg16 or simclr)
and compute embedding & cosine distance matrices that are going to be used 
for further analysis.

TODO: grab_activation not working for coarsegrain generators yet.
"""   

def execute(config,
            compute_semantic_activation=True,
            compute_distance_matrices=True):
    part = 'val_white'
    # dfs = ['amphibian', 'bird', 'fish', 'primate', 'reptile']
    dfs = []
    lossWs = [0, 0.1, 1, 2, 3, 5, 7, 10]
    # -------------------------------------------
    for lossW in lossWs:
        
        if len(dfs) == 0:
            if compute_semantic_activation:
            # get trained model intercepted as `semantic_layer`
                model = ready_model(config=config, 
                                    lossW=lossW)
                grab_activations(model=model, 
                                part=part, 
                                config=config,
                                lossW=lossW)
            if compute_distance_matrices:
                embedding_n_distance_matrices(
                                config=config, 
                                lossW=lossW,
                                part=part, 
                                lang_model=True, 
                                useVGG=False, 
                                bert=False)
        else:
            original_lossW = lossW
            for df in dfs:
                lossW = f'{original_lossW}-sup={df}'

                if compute_semantic_activation:
                    # get trained model intercepted as `semantic_layer`
                    model = ready_model_simclr(config=config, 
                                                lossW=lossW)

                    grab_activations(model=model, 
                                    part=part, 
                                    config=config,
                                    lossW=lossW)

                if compute_distance_matrices:
                    embedding_n_distance_matrices(
                                    config=config, 
                                    lossW=lossW,
                                    part=part, 
                                    lang_model=True, 
                                    useVGG=False, 
                                    bert=False)
    

def grab_activations(model, part, config, lossW):
    """
    Purpose:
    --------
    Receive a loaded model, run each class at a time.
    1. Compute the output matrix which is in (768,) for each image. 
    2. Compute the average 768 vector for each class, and save this vector.

    inputs:
    ------
        model: a trained model
        part: default val_white, can be either train/val/val_white
        config: ..
        lossW: weight on the discrete term.
    """
    # config info
    config_version = config['config_version']
    generator_type = config['generator_type']

    # test data
    wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')
    directory = data_directory(part=part)
    wnids, indices, categories = load_classes(num_classes=1000, df='ranked')

    # one class at a time,
    # each class, final result is an average 768 vector
    for i in range(len(wnids)):
        wnid = wnids[i]
        category = categories[i]

        # check if some classes have been computed so can skip
        save_path = f'resources_{part}/_computed_activations/{config_version}/lossW={lossW}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        file2save = os.path.join(save_path, f'{category}.npy')
        if not os.path.exists(file2save):

            if generator_type == 'simclr_finegrain':
                preprocessing_function = None
                simclr_range = True

            elif generator_type == 'vgg16_finegrain':
                preprocessing_function = preprocess_input
                simclr_range = False                

            elif generator_type == 'vgg16_coarsegrain':
                # TODO. Right now data_generator does not support
                # coarsegrain at all.
                NotImplementedError()
                
            gen, steps = data_generator(
                                directory=directory,
                                classes=[wnid],
                                batch_size=128,
                                seed=42,
                                shuffle=True,
                                subset=None,
                                validation_split=0,
                                class_mode='sparse',
                                target_size=(224, 224),
                                preprocessing_function=preprocessing_function,
                                horizontal_flip=False,
                                wordvec_mtx=None,
                                simclr_range=simclr_range,
                                simclr_augment=False)

            # (N, 768)
            proba = model.predict(gen, steps, verbose=1, workers=3)

            # (768,)
            avg_vec = np.mean(proba, axis=0)
            assert avg_vec.shape == (768,)

            # save avg vec
            np.save(file2save, avg_vec)
            print(f'[Check]: saved {file2save}.')


def embedding_n_distance_matrices(config, 
                                  lossW, 
                                  part, 
                                  lang_model=True, 
                                  useVGG=False, 
                                  bert=False):
    """
    Purpose:
    --------
        Given a model,
        compute an embedding and a distance matrix for targeted activations.
        This is going to save the matrices to make RSA faster.

    returns:
    --------
        the saved result is the entire distance matrix (symmetric)
        and the embedding matrix
    """
    config_version = config['config_version']

    # load bert embedding matrix
    if bert:
        embed_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')
        fname = 'bert'

    # other than bert, get embedding matrix for vgg or language model
    else:
        _, _, categories = load_classes(num_classes=1000, df='ranked')
        embed_mtx = []

        for category in categories:
            if useVGG:
                avg_vec = np.load(f'resources_{part}/_computed_activations/block4_pool=vgg16/{category}.npy')
                fname = 'block4_pool'
            elif lang_model:
                avg_vec = np.load(f'resources_{part}/_computed_activations/{config_version}/lossW={lossW}/{category}.npy')
                fname = f'lossW={lossW}'
            embed_mtx.append(avg_vec)

    # first save embedding matrix (N, D)
    X = np.array(embed_mtx)
    print(f'fname={fname}, X.shape = ', X.shape)

    save_path = f'resources_{part}/_embedding_matrices/{config_version}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, f'{fname}.npy'), X)
    print('embedding matrix saved.\n\n')

    # second save cosine distance matrix (N, N)
    disMtx = cosine_distances(X)
    print(f'fname={fname}, disMtx.shape = ', disMtx.shape)
    # save based on fname
    save_path = f'resources_{part}/_cosine_dist_matrices/{config_version}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, f'{fname}.npy'), disMtx)
    print('cosine distance matrix saved.\n\n')

    # third save L2 distance matrix (N, N)
    disMtx = distance_matrix(X, X)
    print(f'fname={fname}, disMtx.shape = ', disMtx.shape)
    # save based on fname
    save_path = f'resources_{part}/_L2_matrices/{config_version}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, f'{fname}.npy'), disMtx)
    print('L2 distance matrix saved.\n\n')    

