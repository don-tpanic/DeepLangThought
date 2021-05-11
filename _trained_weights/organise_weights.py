# organise model weights into structures
import os
import numpy as np
import subprocess


lossWs = [0, 0.1, 1, 2, 3, 5, 7, 10]
version = '11-11-20'
config_version = 'vgg16_finegrain_v1.1.run1'
sup = None
bert_random = False

if bert_random:
    version = '11-11-20-random'

for lossW in lossWs:
    if sup is not None:
        semantic = f'semantic_weights-{version}-lr=3e-05-lossW={lossW}-sup={sup}.pkl'  
        discrete = f'discrete_weights-{version}-lr=3e-05-lossW={lossW}-sup={sup}.pkl'  
        w2_dense_0 = f'w2_dense_0-{version}-lr=3e-05-lossW={lossW}-sup={sup}.pkl'
        w2_dense_1 = f'w2_dense_1-{version}-lr=3e-05-lossW={lossW}-sup={sup}.pkl'
    else:
        semantic = f'semantic_weights-{version}-lr=3e-05-lossW={lossW}.pkl'  
        discrete = f'discrete_weights-{version}-lr=3e-05-lossW={lossW}.pkl'  
        w2_dense_0 = f'w2_dense_0-{version}-lr=3e-05-lossW={lossW}.pkl'
        w2_dense_1 = f'w2_dense_1-{version}-lr=3e-05-lossW={lossW}.pkl'
    
    weights = [semantic, discrete, w2_dense_0, w2_dense_1]
    for i in range(len(weights)):
        full_path = os.path.join(config_version, weights[i])
        if not os.path.exists(full_path):
            continue
        else:
            # rename to using config version
            # i=0, semantic
            # i=1, discrete
            # i=2, w2_dense_0
            # i=3, w2_dense_1
            if i == 0:
                new_fname = f'semantic_weights-{config_version}-lossW={lossW}.pkl'  
            elif i == 1:
                new_fname = f'discrete_weights-{config_version}-lossW={lossW}.pkl'  
            elif i == 2:
                new_fname = f'w2_dense_0-{config_version}-lossW={lossW}.pkl'  
            elif i == 3:
                new_fname = f'w2_dense_1-{config_version}-lossW={lossW}.pkl'  
            new_full_path = os.path.join(config_version, new_fname)
            subprocess.run(['mv', full_path, new_full_path])
    print('\n\n')


'''
# random - ind - lossW = 0

# sup - fish - lossW = 0

# sup - reptile - lossW = 0

# sup - reptile - random - lossW = 0

'''





