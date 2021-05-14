import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from TRAIN import train_language_modelWithSuperGroups
from TRAIN import train
from TRAIN.utils.data_utils import load_config


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--label', dest='label')
parser.add_argument('-f', '--frontend', dest='frontend')
parser.add_argument('-v', '--version', dest='version')
args = parser.parse_args()

'''
Example command:
    python META_train.py -l finegrain -f simclr -v v1.1.run1 
'''

if __name__ == '__main__':
    config_version = f'{args.frontend}_{args.label}_{args.version}'
    config = load_config(config_version)
    train.execute(config)

    # if args.label == 'finegrain':
    #     config = load_config('simclr_finegrain_v1.1.run1')
    #     train.execute(config)

    # elif args.run == 'coarsegrain':
    #     train_language_modelWithSuperGroups.execute()
