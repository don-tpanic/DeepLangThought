import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
# from TRAIN import train_language_modelWithSuperGroups
# from TRAIN import train
from TRAIN import train_tfrecords
from TRAIN.utils.data_utils import load_config


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--label', dest='label')
parser.add_argument('-f', '--frontend', dest='frontend')
parser.add_argument('-v', '--version', dest='version')
args = parser.parse_args()

'''
Example command:
    python META_train.py -l finegrain -f simclr -v v2.2.run1 
'''

if __name__ == '__main__':
    config_version = f'{args.frontend}_{args.label}_{args.version}'
    config = load_config(config_version)
    # train.execute(config)
    train_tfrecords.execute(config)