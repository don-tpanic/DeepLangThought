import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from pathlib import Path
import h5py
import numpy as np
import pickle
import pandas as pd
import psiz
import scipy.stats
import tensorflow as tf

# from psiz_imagenet import dimensionality
import utils

fp_project = Path.home() / Path('projects', 'psiz-projects', 'imagenet')
fp_ext_embedding = fp_project / Path('target_models')
fp_strain = fp_project / Path('assets', 'strains', 'real')
fp_emb = fp_strain / Path('emb')
# fp_db_human = fp_strain / Path('dbs', 'db_human.txt')
metric = 'correlation'
# print('\nfp_emb = ', fp_emb, '\n')
'''/home/ken/projects/psiz-projects/imagenet/assets/strains/real/emb'''
# print('\nfp_ext_embedding = ', fp_ext_embedding, '\n')
'''/home/ken/projects/psiz-projects/imagenet/target_models'''

distance_list = ['cosine', 'dot']
model_name_list = ['vgg16']

round_idx = 195
batch_size = 10000
n_stimuli = 50000
subset_idx = None

# NOTE: sumbsample=.01 ~ 10s per matrix, .1 ~150s per matrix
ds_pairs, ds_info = psiz.utils.pairwise_index_dataset(
    n_stimuli, batch_size=batch_size, subsample=.01, seed=252)

# Load ensemble.NOTE: filepath to individual models, no order.
mid_list = ['psiz_models/emb-0-118-9-1', ...]

# PsiZ similarity matrix.
s_psiz, within_rho = utils.ensemble_marginalized_similarity(
    mid_list, ds_pairs, ds_info, compute_within=False, mode='spearman',
    verbose=1)

for model_name in model_name_list:

    fp_ext_embedding_features = 'simclr_reprs_val_5k.pkl'

    # Load embeddings of external model and compute similarities.
    z = pickle.load(open(fp_ext_embedding_features, 'rb'))
    for distance in distance_list:

        s_x = utils.similarity_simple(
            tf.constant(z), ds_pairs, ds_info, similarity=distance,
            verbose=1
        )

        # Evaluate correlation.
        rho, _ = scipy.stats.spearmanr(s_psiz, s_x)
        print(f'distance=[{distance}], rho={rho}')