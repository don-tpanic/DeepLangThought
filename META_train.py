import argparse
from TRAIN import train
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
    python META_train.py -l finegrain -f simclr -v v2.2.run1 -r False

Meaning: train with `finegrain` labels, use `simclr` as front end,
          config version `v2.2.run1` and data not stored as `tfrecords`.
"""

if __name__ == '__main__': 
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_index
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    config_version = f'{args.frontend}_{args.label}_{args.version}'
    config = load_config(config_version)
    
    if args.tfrecords:
        train_tfrecords.execute(config)
    else:
        train.execute(config)
