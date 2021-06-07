import pickle
import scipy.stats as stats

w2_depth = 2
lossW = 2
config_versions = ['simclr_finegrain_v3.1.run2', 
                   'simclr_finegrain_v3.1.run2',
                   ]
layers = ['w2_dense_0', 'w2_dense_1', 'semantic_weights', 'discrete_weights']


for layer in layers:
    print('\n===========================')
    print(f'layer = {layer}')
    print('---------------------------')
    collector = []
    for config_version in config_versions:

        with open(f'_trained_weights/{config_version}/{layer}-{config_version}-lossW={lossW}.pkl', 'rb') as f:
            collector.append(pickle.load(f))

    
    for a in range(len(collector)):
        for b in range(len(collector)):
            if a >= b:
                continue

            kernel1 = collector[a][0].flatten()
            kernel2 = collector[b][0].flatten()
            bias1 = collector[a][1].flatten()
            bias2 = collector[b][1].flatten()

            r_kernel, _ = stats.spearmanr(kernel1, kernel2)
            r_bias, _ = stats.spearmanr(bias1, bias2)
            print(config_versions[a][-4:], 'vs', config_versions[b][-4:])
            print(f'r_kernel = {r_kernel}')
            print(f'r_bias = {r_bias}')
