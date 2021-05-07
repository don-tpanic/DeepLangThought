import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import yaml

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from TRAIN.utils.data_utils import load_classes, data_directory
from keras_custom.models.language_model import lang_model, lang_model_contrastive
from keras_custom.generators.generator_wrappers import lang_gen, simclr_gen

"""
Compare how the semantic representations 
from VGG16(fc2) and SIMCLRv2(final_avg_pool) differ.

We use validation set to save compute.
"""

def execute():
    get_model_reprs()
    compute_rsa()


def get_model_reprs():
    # vgg model
    vgg_model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    vgg_model.load_weights('VGG16_finetuned_fullmodelWeights.h5')
    print('[Check] Loaded in fine tuned VGG16 weights.')

    penult_output = vgg_model.get_layer('fc2').output
    vgg_model = Model(inputs=vgg_model.input, outputs=penult_output)
    vgg_model.summary()

    vgg_gen, _ = lang_gen(
                    data_directory(part='val'),
                    classes=None,
                    batch_size=128,
                    seed=42,
                    shuffle=False,
                    subset='validation',
                    validation_split=0.2,
                    class_mode='categorical',
                    target_size=(224, 224),
                    preprocessing_function=preprocess_input,
                    horizontal_flip=False, 
                    wordvec_mtx=np.load('data_local/imagenet2vec/imagenet2vec_1k.npy'))

    vgg_reprs = vgg_model.predict(vgg_gen, verbose=1)
    np.save('vgg_reprs.npy', vgg_reprs)

    # simclr model
    simclr_model = lang_model_contrastive(config=None)
    # simclr_model.compile(tf.keras.optimizers.Adam(), loss=['mse'])
    gen, _ = simclr_gen(
                    data_directory(part='val'),
                    classes=None,
                    batch_size=128,
                    seed=42,
                    shuffle=False,
                    subset='validation',
                    validation_split=0.2,
                    class_mode='categorical',
                    target_size=(224, 224),
                    preprocessing_function=None,
                    horizontal_flip=False, 
                    wordvec_mtx=np.load('data_local/imagenet2vec/imagenet2vec_1k.npy'))
    simclr_reprs = simclr_model.predict(gen, verbose=1)
    np.save('simclr_reprs.npy', simclr_reprs)


def compute_rsa():
    from scipy.stats import spearmanr
    from sklearn.metrics.pairwise import cosine_distances

    emb1 = np.load('vgg_reprs.npy')
    disMtx1 = cosine_distances(emb1)
    print(f'[Check] computed distance mtx 1, shape={disMtx1.shape}')

    emb2 = np.load('simclr_reprs.npy')
    disMtx2 = cosine_distances(emb2)
    print(f'[Check] computed distance mtx 2, shape={disMtx2.shape}')

    # k=0 means including the diagonal
    uptri1 = disMtx1[np.triu_indices(disMtx1.shape[0])]
    uptri2 = disMtx2[np.triu_indices(disMtx2.shape[0])]
    print('[Check] uptri spearman', spearmanr(uptri1, uptri2))


if __name__ == '__main__':
    execute()
