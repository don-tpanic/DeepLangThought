import numpy as np
import pandas as pd
import socket
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import preprocess_input
from keras_custom.generators.generator_wrappers import simclr_gen, lang_gen, sup_gen


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


def train_n_val_data_gen(config, subset, bert_random=False, generator_type='simclr'):
    """
    Purpose:
    --------
        Return a generator can be used for one of the following tasks
            1. VGG front end with either finegrain or coarsegrain labels
            2. simclr front end with either finegrain or coarsegrain labels
    
    inputs:
    -------
        subset: training or validation
        bert_random: True or False
        generator_type: simclr 
        # TODO: this needs regrouped.
    """
    # data generators
    directory = data_directory(part='train')  # default is train, use val only for debug
    if not bert_random:
        wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')
        print('[Check] Using regular BERT...\n')
    else:
        wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k_random98.npy')
        print('[Check] Using random BERT 98...\n')
    
    if generator_type == 'simclr':
        generator = simclr_gen
        preprocessing_function = None
    elif generator_type == 'finegrain':
        generator = lang_gen
        preprocessing_function = preprocess_input
    elif generator_type == 'coarsegrain':
        generator = sup_gen
        preprocessing_function = preprocess_input

    gen, steps = generator(directory=directory,
                           classes=None,
                           batch_size=config['batch_size'],
                           seed=config['generator_seed'],
                           shuffle=True,
                           subset=subset,
                           validation_split=config['validation_split'],
                           class_mode='categorical',
                           target_size=(224, 224),
                           preprocessing_function=preprocessing_function,
                           horizontal_flip=True, 
                           wordvec_mtx=wordvec_mtx)
    return gen, steps



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


def data_directory(part='train'):
    """
    Check which server we are on and return the corresponding 
    imagenet data directory.

    part: partition - either train or val or val_white
    """
    hostname = socket.gethostname()
    if hostname == 'oem-Z11PG-D24-Series':
        print('server: scan test server')
        imagenet_dir = f'/home/oem/datasets/ILSVRC/2012/clsloc/{part}'
    else:
        server_num = int(hostname[4:6])
        print(f'server_num = {server_num}')
        if server_num <= 20:
            imagenet_dir = f'/mnt/fast-data{server_num}/datasets/ILSVRC/2012/clsloc/{part}'
        else:
            imagenet_dir = f'/fast-data{server_num}/datasets/ILSVRC/2012/clsloc/{part}'
    return imagenet_dir