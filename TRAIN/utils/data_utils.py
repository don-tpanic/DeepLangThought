import os
import yaml
import socket
import numpy as np
import pandas as pd
import tensorflow as tf


def specific_callbacks(config, lossW):
    """
    Define earlystopping and tensorboard.

    inputs:
    ------- 
        config: ..
        lossW: This is separated from config.
    """
    config_version = config['config_version']
    earlystopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    min_delta=0, 
                    patience=config['patience'], 
                    verbose=2, 
                    mode='min',
                    baseline=None, 
                    restore_best_weights=True
                    )
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f'log/{config_version}_lossW={lossW}')
    return earlystopping, tensorboard


def load_config(config_version):
    with open(os.path.join('configs', f'{config_version}.yaml')) as f:
        config = yaml.safe_load(f)
    print(f'[Check] Loading [{config_version}]')
    return config


def load_classes(num_classes, df='imagenetA'):
    """
    load in all imagenet/imagenetA or other dataframe classes,
    return:
    -------
        n classes of wnids, indices and descriptions
    """
    df = pd.read_csv(f'groupings-csv/{df}_Imagenet.csv',
                     usecols=['wnid', 'idx', 'description'])
    sorted_indices = np.argsort([i for i in df['wnid']])[:num_classes]

    wnids = np.array([i for i in df['wnid']])[sorted_indices]
    indices = np.array([int(i) for i in df['idx']])[sorted_indices]
    descriptions = np.array([i for i in df['description']])[sorted_indices]
    return wnids.tolist(), indices, descriptions


def data_directory(part, front_end, tfrecords=False):
    """
    Check which server we are on and return the corresponding 
    imagenet data directory.

    part: partition - either train or val or val_white
    """
    hostname = socket.gethostname()
    if hostname == 'oem-Z11PG-D24-Series':
        print('server: scan test server')
        data_dir = f'/home/oem/datasets/ILSVRC/2012/clsloc/{part}'
    else:
        server_num = int(hostname[4:6])
        print(f'server_num = {server_num}')
        if server_num <= 20:
            if tfrecords:
                data_dir = f'/mnt/fast-data{server_num}/datasets/ken/{front_end}_reprs/{part}'
            else:
                data_dir = f'/mnt/fast-data{server_num}/datasets/ILSVRC/2012/clsloc/{part}'
        else:
            if tfrecords:
                data_dir = f'/fast-data{server_num}/datasets/ken/{front_end}_reprs/{part}'
            else:
                data_dir = f'/fast-data{server_num}/datasets/ILSVRC/2012/clsloc/{part}'
    return data_dir

