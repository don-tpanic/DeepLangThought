import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from pathlib import Path
import h5py
import numpy as np
import pickle
import pandas as pd
import psiz
import scipy.stats
import tensorflow as tf
import utils

metric = 'correlation'
distance_list = ['cosine', 'dot']

round_idx = 195
batch_size = 10000
n_stimuli = 50000
subset_idx = None

# NOTE: sumbsample=.01 ~ 10s per matrix, .1 ~150s per matrix
ds_pairs, ds_info = psiz.utils.pairwise_index_dataset(
    n_stimuli, batch_size=batch_size, subsample=.1, seed=252)

# Load ensemble.NOTE: filepath to individual models, no order.
mid_list = ['psiz_models/emb-0-195-4-0', 
            'psiz_models/emb-0-195-4-1',
            'psiz_models/emb-0-195-4-2']

# PsiZ similarity matrix.
s_psiz, within_rho = utils.ensemble_marginalized_similarity(
    mid_list, ds_pairs, ds_info, compute_within=False, mode='spearman',
    verbose=1)


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