import os
import numpy as np
import matplotlib.pyplot as plt
from seaborn import violinplot
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras_custom.generators.generator_wrappers import simclr_gen, simple_generator
from EVAL.utils.data_utils import data_directory, load_classes, load_config
from EVAL.utils.model_utils import ready_model_simclr

"""
Use methods such as tNSE or RSA
    to compare if the labelling effect makes sense.

To do that we want to grab activations from the semantic layer
in both the semantic model and the discrete model.
"""   

def execute(compute_semantic_activation=True,
            compute_distance_matrices=True):
    part = 'val_white'
    config = load_config('test_v1')
    config_version = config['config_version']
    lr = config['lr']
    w2_depth = config['w2_depth']
    intersect_layer = 'semantic'
    fname1 = 'bert'
    df = None
    # lossWs = [0, 0.1, 1, 2, 3, 5, 7, 10]
    lossWs = [1]
    # -------------------------------------------
    for lossW in lossWs:

        if df is not None:
            lossW = f'{lossW}-sup={df}'
        run_name = f'{config_version}-lossW={lossW}'

        if compute_semantic_activation:

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
        save_path = f'RESRC_{part}/_computed_activations/{config_version}/lossW={lossW}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        file2save = os.path.join(save_path, f'{category}.npy')
        if not os.path.exists(file2save):

            if generator_type == 'simclr':
                preprocessing_function = None
                simclr_range = True
            elif generator_type == 'vgg_finegrain':
                NotImplementedError()
            elif generator_type == 'vgg_coarsegrain':
                NotImplementedError()
                
            gen, steps = simple_generator(
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
                                simclr_range=simclr_range)
            

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
        compute a embedding and a distance matrix for targeted activations.
        This is going to save the matrices to make RSA faster.

    returns:
    --------
        the saved result is the entire distance matrix (symmetric)
        and the embedding matrix
    """
    config_version = config['version']

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
                avg_vec = np.load(f'RESRC_{part}/_computed_activations/block4_pool=vgg16/{category}.npy')
                fname = 'block4_pool'
            elif lang_model:
                avg_vec = np.load(f'RESRC_{part}/_computed_activations/{version}/lossW={lossW}/{category}.npy')
                fname = f'{config_version}-lossW={lossW}'
            embed_mtx.append(avg_vec)

    # first save embedding matrix (N, D)
    X = np.array(embed_mtx)
    print(f'fname={fname}, X.shape = ', X.shape)
    np.save(f'RESRC_{part}/_embedding_matrices/{fname}.npy', X)
    print('embedding matrix saved.')

    # second save cosine distance matrix (N, N)
    disMtx = cosine_distances(X)
    print(f'fname={fname}, disMtx.shape = ', disMtx.shape)
    # save based on fname
    np.save(f'RESRC_{part}/_cosine_dist_matrices/{fname}.npy', disMtx)
    print('cosine distance matrix saved.')
    

