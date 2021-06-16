import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import psiz
import pickle
import numpy as np 
import tensorflow as tf

from keras_custom.generators import simclr_preprocessing


"""
To perform big correlation between simclr representations 
and psiz embeddings, here we need to get all the simclr
output layer representations that correspond to the 
50,000 validation set images.

Note, we use `catalog.hdf5` to extract our images as that is 
the order used by https://arxiv.org/abs/2011.11015
"""

def execute():
    model = load_model()

    # image catalog indexed according to psiz catalog
    catalog = psiz.catalog.load_catalog('corr_analysis/psiz_models/catalog.hdf5')
    parent_path = '/fast-data21/datasets/ILSVRC/2012/clsloc/val'
    filepaths = [os.path.join(parent_path, child_path) for child_path in catalog.filepath()]

    collector = np.empty((2048, ))
    for filepath in filepaths:
        print(filepath)
        # load, preprocess image and load into simclr.
        x = tf.keras.preprocessing.image.load_img(filepath)
        x = tf.keras.preprocessing.image.img_to_array(x)
        x = tf.convert_to_tensor(x, dtype=tf.uint8)
        x = simclr_preprocessing._preprocess(x, is_training=False)
        x = tf.reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]])

        # one image repr (2048,)
        x = model.predict(x)[0]  # otherwise len(x)=1
        collector = np.vstack((collector, x))

    # remove the first row
    collector = collector[1:, :]
    print('collector.shape = ', collector.shape)
    with open('corr_analysis/simclr_reprs_val_5k.pkl', 'wb') as f:
        pickle.dump(collector, f)
    print('[Check] pickled')


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




if __name__ == '__main__':
    execute()