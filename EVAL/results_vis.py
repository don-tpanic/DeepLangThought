import os
import numpy as np
import matplotlib.pyplot as plt
from seaborn import violinplot
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
#from matplotlib import rc
#rc('text', usetex=True)

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras_custom.generators.generator_wrappers import lang_gen, simple_generator
from EVAL.utils.data_utils import data_directory, load_classes
from EVAL.utils.model_utils import ready_model

"""
Plotting final results from all evaluations done in
`compare_semantic_activations.py`

Note there is absolutely no evaluations done in this
script, all processing related to versions of the models
are done already. All plotting functions are model-agnostic.
"""

def eval1_similarity_correlation(lossWs, mtx_type='cosine_sim', part='val_white', version='11-11-20'):
    print(f'mtx type = {mtx_type}')

    true_bert = np.load(f'RESRC_{part}/_{mtx_type}_matrices/bert.npy')
    correlations = []
    fig, ax = plt.subplots()
    for lossW in lossWs:
        inferred = np.load(f'RESRC_{part}/_{mtx_type}_matrices/version={version}-lossW={lossW}.npy')
        assert true_bert.shape == inferred.shape

        if mtx_type == 'distance' or mtx_type == 'cosine_sim':
            uptri1 = true_bert[np.triu_indices(true_bert.shape[0])]
            uptri2 = inferred[np.triu_indices(inferred.shape[0])]
            print('uptri spearman', spearmanr(uptri1, uptri2))
            correlations.append(spearmanr(uptri1, uptri2)[0])
    
    ax.scatter(lossWs, correlations)
    ax.set_xlabel(r'loss levels ($\beta$)')
    ax.set_ylabel('spearman correlations')
    plt.grid(True)
    plt.savefig('RESULTS/submission/eval1_similarity_correlation.pdf')
    print('plotted.')


def eval2_indClass_distance(lossWs, part='val_white', version='11-11-20'):
    wnids, indices, categories = load_classes(num_classes=1000, df='ranked')
    fig, ax = plt.subplots()
    all_uptris = []
    for i in range(len(lossWs)):
        lossW = lossWs[i]
        distMtx = np.load(f'RESRC_{part}/_distance_matrices/version={version}-lossW={lossW}.npy')      
        subMtx = distMtx[indices, :][:, indices]
        subMtx_uptri = subMtx[np.triu_indices(subMtx.shape[0])]

        all_uptris.append(subMtx_uptri)

    violinplot(data=all_uptris, cut=-5, linewidth=.8, gridsize=300)
    ax.set_xlabel(r'loss levels ($\beta$)')
    ax.set_xticks(range(len(lossWs)))
    ax.set_xticklabels(lossWs)
    ax.set_ylabel('pairwise class distance')
    plt.savefig(f'RESULTS/submission/eval2_indDistance.pdf')
    print('plotted.')


def eval3_supClass_distance_v1(lossWs, part='val_white', version='27-7-20'):
    """
    Plot one sup at once. v2 plots all sup at once and compute average
    """
    #################
    num_classes = 129
    df = 'canidae'
    #################
    wnids, indices, categories = load_classes(num_classes=num_classes, df=df)
    fig, ax = plt.subplots()
    ratios = []    # ratio btw dog2dog and dog2rest
    for i in range(len(lossWs)):
        lossW = lossWs[i]
        # the entire 1k*1k matrix
        distMtx = np.load(f'RESRC_{part}/_distance_matrices/version={version}-lossW={lossW}-sup={df}.npy')
        #print('WARNING: regular model used on superordinates!!')
        # the dogs matrix      
        subMtx = distMtx[indices, :][:, indices]
        # the uptri of dogs matrix
        subMtx_uptri = subMtx[np.triu_indices(subMtx.shape[0])]
        # what we already know about dog vs dog
        mean_dist = np.mean(subMtx_uptri)
        std_dist = np.std(subMtx_uptri)
        # ------------------------------------------------------
        # new stuff: dog vs rest
        nonDog_indices = [i for i in range(1000) if i not in indices]
        # shape = (129, 871)
        dogVSrest_mtx = distMtx[indices, :][:, nonDog_indices]
        dogVSrest_mean_dist = np.mean(dogVSrest_mtx)
        dogVSrest_std_dist = np.std(dogVSrest_mtx)

        ratio = mean_dist / dogVSrest_mean_dist
        ratios.append(ratio)
        print(f'dog2dog = {mean_dist}(std = {std_dist}')
        print(f'dog2rest = {dogVSrest_mean_dist}(std = {dogVSrest_std_dist}')

    ax.scatter(lossWs, ratios)
    ax.set_xlabel('loss levels')
    ax.set_ylabel('relative distance')
    plt.grid(True)
    plt.savefig(f'RESULTS/submission/supDistance_ratio-{df}.pdf')
    print('plotted.')


def eval3_supClass_distance_v2(lossWs, part='val_white', version='11-11-20'):
    """
    Plot all sup at once.
    """
    dfs = ['reptile', 'amphibian', 'fish', 'bird', 'canidae', 'primate']    
    markers = ['*', '<', 'o', '^', '>']

    all_ratios = np.zeros((len(dfs), len(lossWs)))
    fig, ax = plt.subplots()
    for z in range(len(dfs)):
        df = dfs[z]
        print(f'processing {df}...')
        wnids, indices, categories = load_classes(num_classes=1000, df=df)  # num_classes doesn't matter cuz subset<1000
        ratios = []    # ratio btw dog2dog and dog2rest
        for i in range(len(lossWs)):
            lossW = lossWs[i]
            # the entire 1k*1k matrix
            distMtx = np.load(f'RESRC_{part}/_distance_matrices/version={version}-lossW={lossW}-sup={df}.npy')
            # the dogs matrix      
            subMtx = distMtx[indices, :][:, indices]
            # the uptri of dogs matrix
            subMtx_uptri = subMtx[np.triu_indices(subMtx.shape[0])]
            # what we already know about dog vs dog
            mean_dist = np.mean(subMtx_uptri)
            std_dist = np.std(subMtx_uptri)
            # ------------------------------------------------------
            # new stuff: dog vs rest
            nonDog_indices = [i for i in range(1000) if i not in indices]
            # shape = (129, 871)
            dogVSrest_mtx = distMtx[indices, :][:, nonDog_indices]
            dogVSrest_mean_dist = np.mean(dogVSrest_mtx)
            dogVSrest_std_dist = np.std(dogVSrest_mtx)
            ratio = mean_dist / dogVSrest_mean_dist
            ratios.append(ratio)

        if df == 'canidae':
            df = 'dog'
        ax.scatter(lossWs, ratios, label=f'{df}', marker=markers[z])
        all_ratios[z, :] = ratios
    
    print('all_ratios.shape = ', all_ratios.shape)
    average_ratios = np.mean(all_ratios, axis=0)
    ax.plot(lossWs, average_ratios, label='average')
    ax.set_xlabel(r'loss levels ($\beta$)')
    ax.set_ylabel('distance ratio')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'RESULTS/submission/eval3_supDistance_avg_ratios.pdf')


def execute():
    lossWs = [0, 0.1, 1, 2, 3, 5, 7, 10]
    #eval1_similarity_correlation(lossWs)
    #eval2_indClass_distance(lossWs)
    eval3_supClass_distance_v2(lossWs)
    


