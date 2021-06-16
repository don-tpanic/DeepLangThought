# -*- coding: utf-8 -*-
# Copyright 2020 Brett D. Roads. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Evaluate correlation of models with ImageNet-HSJ."""

import os
from pathlib import Path

import h5py
import numpy as np
import pickle
import pandas as pd
import psiz
import scipy.stats
from sklearn.preprocessing import normalize
import tensorflow as tf

# from psiz_imagenet import dimensionality
# from psiz_imagenet import utils
# import psiz_imagenet.database as db
import utils

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


if __name__ == "__main__":
    # Settings.
    fp_project = Path.home() / Path('projects', 'psiz-projects', 'imagenet')
    fp_ext_embedding = fp_project / Path('target_models')
    # Strain-specific paths.
    fp_strain = fp_project / Path('assets', 'strains', 'real')
    fp_fit_log = fp_strain / Path('dbs', 'db_fit.txt')
    fp_db_dim = fp_strain / Path('dbs', 'db_dim.txt')
    fp_emb = fp_strain / Path('emb')
    fp_db_human = fp_strain / Path('dbs', 'db_human.txt')
    metric = 'correlation'
    # version_list = ['0.1', '0.2']
    version_list = ['0.3']
    print('\nfp_emb = ', fp_emb, '\n')

    # Setup catalog.
    fp_catalog = fp_project / Path('assets', 'stimuli', 'catalog.hdf5')
    common_path = fp_project / Path('assets', 'stimuli', 'resized')
    catalog = psiz.catalog.load_catalog(fp_catalog)

    # TODO load db_evo instead.
    # Load fit log and dimension database.
    df_fit = db.load_db(fp_fit_log)
    df_dim = db.load_db(fp_db_dim)

    # Filter down to best model for each round.
    df_best = db.best_only(df_fit, df_dim)

    distance_list = ['cosine', 'dot']

    model_name_list = [
        'vgg16', 'vgg19', 'resnet50', 'resnet101', 'resnet152',
        'resnet50v2', 'xception', 'inceptionv3', 'InceptionResNetV2',
        'densenet121', 'MobileNet', 'dc-vgg16'
    ]

    for version in version_list:
        if version == '0.1':
            # Seed settings.
            round_idx = 118
            # Seed settings.
            batch_size = 10000
            n_stimuli = 1000
            subset_idx = np.arange(n_stimuli)
            # Create all pairs.
            ds_pairs, ds_info = utils.pairwise_index_dataset(
                n_stimuli, batch_size=batch_size
            )
        elif version == '0.2':
            # Full validation set.
            round_idx = 195
            batch_size = 10000
            n_stimuli = 50000
            subset_idx = None
            # NOTE: sumbsample=.01 ~ 10s per matrix, .1 ~150s per matrix
            ds_pairs, ds_info = utils.pairwise_index_dataset(
                n_stimuli, batch_size=batch_size, subsample=.1, seed=252
            )
        elif version == '0.3':
            # Full validation set.
            round_idx = 220
            batch_size = 10000
            n_stimuli = 50000
            subset_idx = None
            # NOTE: sumbsample=.01 ~ 10s per matrix, .1 ~150s per matrix
            ds_pairs, ds_info = utils.pairwise_index_dataset(
                n_stimuli, batch_size=batch_size, subsample=.1, seed=252
            )

        # Load ensemble.
        mid_list, _ = utils.assemble_mid_list(df_best, round_idx, fp_emb)

        # PsiZ similarity matrix.
        s_psiz, within_rho = utils.ensemble_marginalized_similarity(
            mid_list, ds_pairs, ds_info, compute_within=True, mode='spearman',
            verbose=1
        )
        print('    Within rho_s: {0:.2f}'.format(within_rho))
        # Psiz baseline.
        # ID data.
        id_data = {
            'model': 'psiz', 'input_id': round_idx,
            'distance': 'exp', 'metric': metric
        }
        # Associated data.
        assoc_data = {'score': within_rho}
        df_human = db.load_db(fp_db_human)
        df_human = db.update_one(df_human, id_data, assoc_data)
        db.save_db(df_human, fp_db_human)

        for model_name in model_name_list:

            fp_ext_embedding_features = fp_ext_embedding / Path(
                'emb_{0}.p'.format(model_name)
            )
            # Load embeddings of external model and compute similarities.
            z = pickle.load(open(fp_ext_embedding_features, 'rb'))

            # TODO delete
            # DeepCluster VGG16.
            # model_name = 'dc-vgg16'
            # distance = 'cosine'
            # # distance = 'exp'
            # fp_ext_embedding_features = fp_ext_embedding / Path(
            #     'emb_{0}.p'.format(model_name)
            # )
            # # # Load embeddings of external model and compute similarities.
            # z = pickle.load(open(fp_ext_embedding_features, 'rb'))

            for distance in distance_list:

                s_x = utils.similarity_simple(
                    tf.constant(z), ds_pairs, ds_info, similarity=distance,
                    verbose=1
                )

                # Evaluate correlation.
                rho, _ = scipy.stats.spearmanr(s_psiz, s_x)

                # Update database.
                # ID data.
                id_data = {
                    'model': model_name, 'input_id': round_idx,
                    'distance': distance, 'metric': metric
                }
                # Associated data.
                assoc_data = {'score': rho}
                df_human = db.load_db(fp_db_human)
                df_human = db.update_one(df_human, id_data, assoc_data)
                db.save_db(df_human, fp_db_human)
