import os
import numpy as np
import matplotlib.pyplot as plt
from seaborn import violinplot
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams.update({'font.size': 16})

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

def eval1_similarity_correlation(lossWs, mtx_type='cosine_sim', part='val_white', version='11-11-20', sup=None):
    print(f'mtx type = {mtx_type}')

    true_bert = np.load(f'RESRC_{part}/_{mtx_type}_matrices/bert.npy')
    correlations = []
    fig, ax = plt.subplots()
    for lossW in lossWs:
        if sup is None:
            inferred = np.load(f'RESRC_{part}/_{mtx_type}_matrices/version={version}-lossW={lossW}.npy')
        else:
            inferred = np.load(f'RESRC_{part}/_{mtx_type}_matrices/version={version}-lossW={lossW}-sup={sup}.npy')
        assert true_bert.shape == inferred.shape

        if mtx_type in ['distance', 'cosine_sim', 'cosine_dist']:
            uptri1 = true_bert[np.triu_indices(true_bert.shape[0])]
            uptri2 = inferred[np.triu_indices(inferred.shape[0])]
            print('uptri spearman', spearmanr(uptri1, uptri2))
            correlations.append(spearmanr(uptri1, uptri2)[0])
    
    ax.scatter(lossWs, correlations)
    ax.set_xlabel(r'loss levels ($\beta$)')
    ax.set_ylabel('spearman correlations')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('RESULTS/submission/Exp1_fig1.pdf')
    print('plotted.')


def eval2_indClass_distance(lossWs, part='val_white', version='11-11-20'):
    wnids, indices, categories = load_classes(num_classes=1000, df='ranked')
    fig, ax = plt.subplots()
    all_uptris = []

    # first the reference distance distribution
    # distMtx = np.load(f'RESRC_{part}/_distance_matrices/bert.npy')      
    # subMtx = distMtx[indices, :][:, indices]
    # subMtx_uptri = subMtx[np.triu_indices(subMtx.shape[0])]
    # all_uptris.append(subMtx_uptri)

    for i in range(len(lossWs)):
        lossW = lossWs[i]
        distMtx = np.load(f'RESRC_{part}/_distance_matrices/version={version}-lossW={lossW}.npy')      
        subMtx = distMtx[indices, :][:, indices]
        subMtx_uptri = subMtx[np.triu_indices(subMtx.shape[0])]

        all_uptris.append(subMtx_uptri)

    violinplot(data=all_uptris, cut=-5, linewidth=.8, gridsize=300)
    ax.set_xlabel(r'loss levels ($\beta$)')
    ax.set_xticks(range(len(lossWs)+1))
    # ax.set_xticklabels(['reference'] + lossWs)
    ax.set_ylabel('pairwise class distance')
    plt.savefig(f'RESULTS/submission/Exp1_fig2.pdf')
    # plt.tight_layout()
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
    ax.set_xlabel(r'loss levels ($\beta$)')
    ax.set_ylabel('relative distance')
    plt.grid(True)
    plt.savefig(f'RESULTS/submission/supDistance_ratio-{df}.pdf')
    print('plotted.')


def eval3_supClass_distance_v2(lossWs, part='val_white', version='11-11-20'):
    """
    Plot all sup at once.
    """
    dfs = ['reptile', 'amphibian', 'primate', 'bird', 'canidae']    
    markers = ['*', '<', 'o', '^', '>']

    all_ratios = np.zeros((len(dfs), len(lossWs)))
    fig, ax = plt.subplots()
    for z in range(len(dfs)):
        df = dfs[z]
        if df == 'canidae':
            version = '27-7-20'
        else:
            version = '11-11-20'
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
    plt.tight_layout()
    plt.savefig(f'RESULTS/submission/Exp2_fig2.pdf')


def eval1_similarity_correlation_v2(lossWs, mtx_type='cosine_sim', part='val_white', version='11-11-20'):
    """
    Added experiment for plotting RSA results for superordinate labels
    We average over five superordinates and plot the average correlations.
    """
    print(f'mtx type = {mtx_type}')

    true_bert = np.load(f'RESRC_{part}/_{mtx_type}_matrices/bert.npy')

    sups = ['reptile', 'amphibian', 'primate', 'bird', 'canidae']
    markers = ['*', '<', 'o', '^', '>']
    fig, ax = plt.subplots()

    accu_correlations = np.empty((len(sups), len(lossWs)))
    for i in range(len(sups)):

        sup = sups[i]
        correlations = []

        for j in range(len(lossWs)):

            lossW = lossWs[j]

            if sup is None:
                inferred = np.load(f'RESRC_{part}/_{mtx_type}_matrices/version={version}-lossW={lossW}.npy')
            else:
                # ad hoc where we use the old version of canidae
                # 11-11-20 does not exsit.
                if sup == 'canidae':
                    version = '27-7-20'
                inferred = np.load(f'RESRC_{part}/_{mtx_type}_matrices/version={version}-lossW={lossW}-sup={sup}.npy')
            assert true_bert.shape == inferred.shape

            if mtx_type in ['distance', 'cosine_sim', 'cosine_dist']:
                uptri1 = true_bert[np.triu_indices(true_bert.shape[0])]
                uptri2 = inferred[np.triu_indices(inferred.shape[0])]
                print('uptri spearman', spearmanr(uptri1, uptri2))

                accu_correlations[i, j] = spearmanr(uptri1, uptri2)[0]
                correlations.append(spearmanr(uptri1, uptri2)[0])
        if sups[i] == 'canidae':
            sup = 'dog'
        ax.scatter(lossWs, correlations, marker=markers[i], label=sups[i])

    ax.plot(lossWs, np.mean(accu_correlations, axis=0), label='average')
    ax.set_xlabel(r'loss levels ($\beta$)')
    ax.set_ylabel('spearman correlations')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('RESULTS/submission/Exp2_fig1.pdf')
    print('plotted.')



def execute():
    lossWs = [0, 0.1, 1, 2, 3, 5, 7, 10]
    # eval1_similarity_correlation(lossWs, mtx_type='cosine_dist')
    # eval2_indClass_distance(lossWs)
    # eval3_supClass_distance_v2(lossWs)

    eval1_similarity_correlation_v2(lossWs, mtx_type='cosine_dist')
    


