import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np 
import tensorflow as tf

from iterate_tfrecords import DirectoryIterator

"""
Data loading script using tfrecords.

We should be able to:
    1. Prepare Dataset object for training and testing
    2. For training, 
        - we need to create training/validation (10%)
        - needs to have shuffling
    3. Finegrain vs coarse labels need to be added correctly.
"""

def prepare_dataset(part='val_white'):
    """
    Purpose:
    --------
        Once all simclr outputs are .tfrecords,
        Convert them into tf.Dataset for fitting.
    """
    
    dataset_iterator = DirectoryIterator(directory=f'simclr_reprs/{part}')
    filepaths = dataset_iterator._filepaths
    labels = dataset_iterator.classes

    # TODO: need to have train/val split.
    # TODO: in iterator, add sup choice so labels returned are sup.

    dataset_x_n_semantics = tf.data.Dataset.from_tensor_slices(filepaths)
    dataset_x_n_semantics = dataset_x_n_semantics.interleave(tf.data.TFRecordDataset)

    # returns dataset that can be trained on.
    # in other words, all files need to be loaded prior 
    # this point.
    return dataset_x_n_semantics.map(read_tfrecord)


def read_tfrecord(serialized_example):
    """
    Purpose:
    --------
        Given one image's 

    inputs:
    -------
        serialized_example: a *.tfrecords file
    """
    feature_description = {
        'x': tf.io.FixedLenFeature((), tf.string),
        'x_length': tf.io.FixedLenFeature((), tf.int64),
        'word_emb': tf.io.FixedLenFeature((), tf.string),
        'word_emb_length': tf.io.FixedLenFeature((), tf.int64),
        'label': tf.io.FixedLenFeature((), tf.string),
        'label_length': tf.io.FixedLenFeature((), tf.int64)
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    # x
    x = tf.io.parse_tensor(example['x'], out_type=float)
    x_length = [example['x_length']]
    x = tf.reshape(x, x_length)

    # semantic vector
    word_emb = tf.io.parse_tensor(example['word_emb'], out_type=float)
    word_emb_length = [example['word_emb_length']]
    word_emb = tf.reshape(word_emb, word_emb_length)  # reshape, so shape becomes known.

    # one hot label
    label = tf.io.parse_tensor(example['label'], out_type=tf.int64)
    label_length = [example['label_length']]
    label = tf.reshape(label, label_length)
    return x, word_emb, label










if __name__ == '__main__':
    pass