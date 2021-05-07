import numpy as np 
import pickle

"""
Utils for saving trained weights.
"""

def save_model_weights(model, config, lossW):
    """save weights including w2 dense, semantic, and discrete"""
    w2_depth = config['w2_depth'] 
    config_version = config['config_version']

    # w2 dense
    for i in range(w2_depth):
        dense_ws = model.get_layer(f'w2_dense_{i}').get_weights()
        with open(f'_trained_weights/w2_dense_{i}-{config_version}-lossW={lossW}.pkl', 'wb') as f:
            pickle.dump(dense_ws, f)

    # semantic
    semantic_ws = model.get_layer('semantic_layer').get_weights()
    with open(f'_trained_weights/semantic_weights-{config_version}-lossW={lossW}.pkl', 'wb') as f:
        pickle.dump(semantic_ws, f)

    # save discrete too if w3 were notfrozen
    discrete_weights = model.get_layer('discrete_layer').get_weights()
    with open(f'_trained_weights/discrete_weights-{config_version}-lossW={lossW}.pkl', 'wb') as f:
        pickle.dump(discrete_weights, f)
    print('[Check] All weights saved.')
