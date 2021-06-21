import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from TRAIN import train_tfrecords
from TRAIN.utils.data_utils import load_config


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
parser.add_argument('-r', '--tfrecords', dest='tfrecords', type=str2bool)
parser.add_argument('-gpu', '--gpu', dest='gpu_index')
args = parser.parse_args()

"""
Example command:
    python META_train.py -l coarsegrain -f simclr -v v3.1.run12 -r True -gpu 0

Meaning:
    - train with `coarsegrain` labels,
    - use `simclr` as front end,
    - config version `v3.1.run12`
    - data pipeline uses tfrecords
    - use gpu 0 
"""

if __name__ == '__main__': 
    
    os.environ["CUDA_VISIBLE_DEVICES"]= f"{args.gpu_index}"

    config_version = f'{args.frontend}_{args.label}_{args.version}'
    config = load_config(config_version)
    
    if args.tfrecords:
        train_tfrecords.execute(config)