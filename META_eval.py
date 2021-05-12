import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
# from EVAL import compare_semantic_activations
from EVAL import compute_activations_n_matrices
# from EVAL import results_vis
from EVAL.utils.data_utils import load_config
from EVAL import check_trained_simclr_acc


def str2bool(v):
    """
    Purpose:
    --------
        Such that parser returns boolean as input to function.
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--label', dest='label')
parser.add_argument('-f', '--frontend', dest='frontend')
parser.add_argument('-v', '--version', dest='version')
parser.add_argument('-s', '--semantics', dest='semantics', type=str2bool)
parser.add_argument('-m', '--matrices', dest='matrices', type=str2bool)
parser.add_argument('-a', '--accuracy', dest='accuracy', type=str2bool)
args = parser.parse_args()

'''
Example command:
    python META_eval.py -l finegrain -f simclr -v v1.1.run1 -s True -m True
'''

if __name__ == '__main__':
    
    config_version = f'{args.frontend}_{args.label}_{args.version}'
    config = load_config(config_version)

    if args.semantics is not None:
        print(f'**** Computing intermediate results for plotting ****')
        whether_compute_semantics = args.semantics
        whether_compute_matrices = args.matrices
        compute_activations_n_matrices.execute(config=config, 
                                            compute_semantic_activation=whether_compute_semantics,
                                            compute_distance_matrices=whether_compute_matrices)
    elif args.accuracy is True:
        print(f'**** Computing trained model accuracy ****')
        check_trained_simclr_acc.execute(config)



    # if args.run == 'eval':
    #     print('***** evaluating intermediate results used for plotting *****')
    #     # compare_semantic_activations.execute()

    #     # TODO. Need user choice to do eval finegrain or coarsegrain.
    #     compute_activations_n_matrices.execute()

    # TODO. needs update
    # elif args.run == 'plot':
    #     print('***** plotting figures *****')
    #     results_vis.execute()
