import os
import numpy as np 
import pickle

"""
Utils for saving trained weights.
"""

def save_model_weights(model, config, lossW):
    """save weights including w2 dense, semantic, and discrete"""
    w2_depth = config['w2_depth'] 
    config_version = config['config_version']

    save_path = f'_trained_weights/{config_version}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # w2 dense
    for i in range(w2_depth):
        dense_ws = model.get_layer(f'w2_dense_{i}').get_weights()
        fname = f'w2_dense_{i}-{config_version}-lossW={lossW}.pkl'
        with open(os.path.join(save_path, fname), 'wb') as f:
            pickle.dump(dense_ws, f)

    # semantic
    semantic_ws = model.get_layer('semantic_layer').get_weights()
    fname = f'semantic_weights-{config_version}-lossW={lossW}.pkl'
    with open(os.path.join(save_path, fname), 'wb') as f:
        pickle.dump(semantic_ws, f)

    # save discrete too if w3 were notfrozen
    discrete_weights = model.get_layer('discrete_layer').get_weights()
    fname = f'discrete_weights-{config_version}-lossW={lossW}.pkl'
    with open(os.path.join(save_path, fname), 'wb') as f:
        pickle.dump(discrete_weights, f)
    print('[Check] All weights saved.')
