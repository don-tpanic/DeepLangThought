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
"""Project specific utility functions."""

import os
from pathlib import Path
import time

import numpy as np
import psiz
import tensorflow as tf


def ensemble_marginalized_similarity(
        mid_list, ds_pairs, ds_info, compute_within=False,
        mode='pearson', n_posterior_sample=300, verbose=0):
    """Ensemble mased similarity matrix."""
    n_model = len(mid_list)
    check_wi_ensemble_r2 = True

    smat_list = []
    for mid in mid_list:
        model = tf.keras.models.load_model(mid.pathname)
        sim_arr = psiz.utils.pairwise_similarity(
            model.stimuli, model.kernel, ds_pairs,
            n_sample=n_posterior_sample, compute_average=True,
            verbose=verbose
        )
        smat_list.append(sim_arr)

    idx_i, idx_j = np.triu_indices(n_model, k=1)
    rho_within = []
    if compute_within:
        for idx_a, idx_b in zip(idx_i, idx_j):
            rho_within.append(
                rho_metric(smat_list[idx_a], smat_list[idx_b], mode=mode)
            )
        rho_within = np.mean(rho_within)

    # Compute ensemble average.
    smat = np.stack(smat_list, axis=0)
    smat = np.mean(smat, axis=0)

    return smat, rho_within


def similarity_simple(z, ds_pairs, ds_info, similarity='cosine', verbose=0):
    """Compute pairwise matrix (unrolled).

    Arguments:
        z: Embedding point-estimate.

    Returns:
        simmat_unr: Unrolled similarity matrix.

    """
    start_s = time.time()
    s_unr = []
    counter = 0
    if verbose > 0:
        progbar = psiz.utils.ProgressBarRe(
            ds_info['n_batch'], prefix='Similarity:', length=50
        )
        progbar.update(0)
    for x_batch in ds_pairs:
        z_0 = tf.gather(z, x_batch[0])
        z_1 = tf.gather(z, x_batch[1])
        if similarity == 'cosine':
            s_unr.append(
                tf.negative(
                    tf.keras.losses.cosine_similarity(
                        z_0, z_1, axis=1
                    )
                )
            )
        elif similarity == 'exp':
            s_unr.append(
                tf.exp(-tf.sqrt(tf.reduce_sum(tf.pow(z_0 - z_1, 2), axis=1)))
            )
        elif similarity == 'dot':
            s_unr.append(
                tf.reduce_sum(z_0 * z_1, axis=1)
            )

        if verbose and (np.mod(counter, 10) == 0):
            progbar.update(counter + 1)
        counter += 1

    duration_s = time.time() - start_s
    if verbose > 0:
        print('  Finished evaluation {0:.2f} s'.format(duration_s))
    start_s = time.time()
    s_unr = tf.concat(s_unr, axis=0).numpy()
    duration_s = time.time() - start_s
    # NOTE: tf.concat is very fast.
    # if verbose > 0:
    #     print('  Finished concat {0:.2f} s'.format(duration_s))

    return s_unr


