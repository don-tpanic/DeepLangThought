import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
# from TRAIN import train_language_modelWithSuperGroups
from TRAIN import train
from TRAIN import train_tfrecords
from TRAIN.utils.data_utils import load_config


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--label', dest='label')
parser.add_argument('-f', '--frontend', dest='frontend')
parser.add_argument('-v', '--version', dest='version')
parser.add_argument('-g', '--gpu', dest='gpu_index')
args = parser.parse_args()

'''
Example command:
    python META_train.py -l finegrain -f simclr -v v2.2.run1 -g 3

Meaning:
    Train with `finegrain` labels, use `simclr` as front end,
    with config `v2.2.run1` and on gpu `3`.
'''

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= f'{args.gpu_index}'

    config_version = f'{args.frontend}_{args.label}_{args.version}'
    config = load_config(config_version)
    train_tfrecords.execute(config)
