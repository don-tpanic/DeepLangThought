import os
import numpy as np
import matplotlib.pyplot as plt
from seaborn import violinplot
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import rc
from EVAL.utils.data_utils import data_directory, load_classes
from EVAL.utils.model_utils import ready_model

"""
Plotting final results from all evaluations done in
`compute_activations_n_matrices.py`

Note there is absolutely no evaluations done in this
script, all processing related to versions of the models
are done already. All plotting functions are model-agnostic.
"""

# rc('text', usetex=True)
plt.rcParams.update({'font.size': 20})



def Exp1_AB(config, lossWs, results_path, part='val_white', sup=None):
    """
    Finegrain experiment figures.
    """
    ##### A ######
    config_version = config['config_version']
    true_bert = np.load(f'resources_{part}/_cosine_dist_matrices/bert.npy')
    correlations = []
    fig, ax = plt.subplots(1, 2, figsize=(15,6))
    for lossW in lossWs:
        if sup is None:
            inferred = np.load(f'resources_{part}/_cosine_dist_matrices/{config_version}/lossW={lossW}.npy')
        else:
            inferred = np.load(f'resources_{part}/_cosine_dist_matrices/{config_version}/lossW={lossW}-sup={sup}.npy')
        assert true_bert.shape == inferred.shape

        uptri1 = true_bert[np.triu_indices(true_bert.shape[0])]
        uptri2 = inferred[np.triu_indices(inferred.shape[0])]
        print(spearmanr(uptri1, uptri2))
        correlations.append(spearmanr(uptri1, uptri2)[0])
    
    ax[0].scatter(np.arange(len(lossWs)), correlations)
    ax[0].set_xticks(np.arange(len(lossWs)))
    ax[0].set_xticklabels(lossWs)
    # ax[0].set_xlabel(r'Labelling pressure ($\beta$)')
    ax[0].set_ylabel('Spearman correlations')
    ax[0].set_title('(A)')
    ax[0].grid(True)

    ###### B ######
    wnids, indices, categories = load_classes(num_classes=1000, df='ranked')
    all_uptris = []
    for i in range(len(lossWs)):
        lossW = lossWs[i]
        distMtx = np.load(f'resources_{part}/_L2_matrices/{config_version}/lossW={lossW}.npy')      
        subMtx = distMtx[indices, :][:, indices]
        subMtx_uptri = subMtx[np.triu_indices(subMtx.shape[0])]

        all_uptris.append(subMtx_uptri)

    violinplot(data=all_uptris, cut=-5, linewidth=.8, gridsize=300)
    # ax[1].set_xlabel(r'Labelling pressure ($\beta$)')
    ax[1].set_xticks(np.arange(len(lossWs)))
    ax[1].set_xticklabels(lossWs)
    ax[1].set_ylabel('Pairwise class distance')
    ax[1].set_title('(B)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'Exp1_AB.pdf'))
    print('[Check] Plotted at:', os.path.join(results_path, 'Exp1_AB.pdf'))


def Exp2_AB(lossWs, mtx_type='cosine_dist', part='val_white', version='11-11-20'):
    """
    """
    print(f'mtx type = {mtx_type}')

    true_bert = np.load(f'RESRC_{part}/_{mtx_type}_matrices/bert.npy')

    sups = ['reptile', 'amphibian', 'primate', 'bird', 'canidae']
    markers = ['*', '<', 'o', '^', '>']
    fig, ax = plt.subplots(1, 2, figsize=(15,6))

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
            ax[0].scatter(np.arange(len(lossWs)), correlations, marker=markers[i], label='dog')
        else:
            ax[0].scatter(np.arange(len(lossWs)), correlations, marker=markers[i], label=sups[i])    
    

    ax[0].plot(np.arange(len(lossWs)), np.mean(accu_correlations, axis=0), label='average')
    ax[0].set_xticks(np.arange(len(lossWs)))
    ax[0].set_xticklabels(lossWs)
    ax[0].set_xlabel(r'Labelling pressure ($\beta$)')
    ax[0].set_ylabel('Spearman correlations')
    ax[0].set_title('(A)')
    ax[0].grid(True)


    print('A is ok')
    ###### B #######
    dfs = ['reptile', 'amphibian', 'primate', 'bird', 'canidae']    
    markers = ['*', '<', 'o', '^', '>']

    all_ratios = np.zeros((len(dfs), len(lossWs)))
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
        ax[1].scatter(np.arange(len(lossWs)), ratios, label=f'{df}', marker=markers[z])
        all_ratios[z, :] = ratios
    
    print('all_ratios.shape = ', all_ratios.shape)
    average_ratios = np.mean(all_ratios, axis=0)
    ax[1].plot(np.arange(len(lossWs)), average_ratios, label='average')
    ax[1].set_xlabel(r'Labelling pressure ($\beta$)')
    ax[1].set_ylabel('Distance ratio')
    ax[1].grid(True)
    ax[1].set_title('(B)')
    ax[1].set_xticks(np.arange(len(lossWs)))
    ax[1].set_xticklabels(lossWs)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(results_path, 'Exp2_AB.pdf'))
    print('[Check] Plotted at:', os.path.join(results_path, 'Exp2_AB.pdf'))


def execute(config):
    lossWs = [0, 0.1, 1, 2, 3, 5, 7, 10]

    config_version = config['config_version']
    results_path = f'RESULTS/revision_1/{config_version}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    Exp1_AB(config, lossWs, results_path)

    # TODO. Update when trainer is ready.
    # Exp2_AB(lossWs)
    


