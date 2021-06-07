import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import psiz
import numpy as np 
import tensorflow as tf

from corr_analysis import data_gen

"""
Compare simclr similarity representations
to psiz.
"""

def load_model(name='simclr'):
    """
    Purpose:
    --------
        Load the pretrained simclr model at `final_avg_pool` layer.
    """
    if name == 'simclr':
        class SimclrFrontEnd(tf.keras.Model):
            def __init__(self):
                """
                Load in the pretrained simclr
                And add the same layers as in previous version.
                """
                super(SimclrFrontEnd, self).__init__()
                self.saved_model = tf.saved_model.load('r50_1x_sk0/saved_model/')

            def call(self, inputs):
                """
                Straightforward feedforward net
                """
                simclr_outputs = self.saved_model(inputs, trainable=False)
                return simclr_outputs['final_avg_pool']
        return SimclrFrontEnd()


def load_data():
    """
    return a generator that loads images in the order as the same
    as in psychological embeddings based on `catalog.hdf5`
    """
    batch_size = 128
    catalog = psiz.catalog.load_catalog('corr_analysis/psiz_models/catalog.hdf5')
    parent_path = '/fast-data21/datasets/ILSVRC/2012/clsloc/val'
    filepaths = [os.path.join(parent_path, child_path) for child_path in catalog.filepath()]
    steps = np.ceil(len(filepaths) / batch_size)
    gen = data_gen.DirectoryIterator(
                batch_size=batch_size,
                shuffle=False,
                preprocessing_function=None,
                simclr_range=True,
                simclr_augment=False,
                filepaths=filepaths)
    return gen, steps


def load_psych_emb(model='emb-0-195-4-0', method='posterior'):


    model = tf.keras.models.load_model(f'corr_analysis/psiz_models/{model}')

    if method == 'posterior':
        z = model.stimuli.embeddings.mode()
    else:
        z = model.stimuli.embeddings
    if model.stimuli.mask_zero:
        z = z[1:]
    print(f'[Check] z.shape = {z.shape}')
    return z
