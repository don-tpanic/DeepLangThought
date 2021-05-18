import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np 
import tensorflow as tf
from functools import partial
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
    # can shuffle out here

    # x
    temp = tf.data.Dataset.from_tensor_slices(filepaths)
    temp = temp.interleave(tf.data.TFRecordDataset)
    dataset_x = temp.map(partial(parse_tfrecord, 'x'))

    # semantics
    dataset_s = temp.map(partial(parse_tfrecord, 'word_emb'))

    # labels
    dataset_y = tf.data.Dataset.from_tensor_slices(labels)

    # For targets, we zip (semantics, labels)
    dataset_target = tf.data.Dataset.zip((dataset_s, dataset_y))

    # For entire, we zip (x, targets)
    dataset = tf.data.Dataset.zip((dataset_x, dataset_target))

    return dataset


def parse_tfrecord(component, serialized_example):
    """
    Purpose:
    --------
        Given one image's

    inputs:
    -------
        serialized_example: a *.tfrecords file
        component: x or word_emb or label, once is it set,
                   we only return Dataset made of one component,
                   not all.

    # NOTE: one trick we apply here is to have one extra argument
            `component` which is not in regular parser. This is for 
            us to extract only partial content later as we want.
            The reason is because, when we parse all components at 
            once, we do not get to control the structure of the Dataset
            output. 

            e.g. if we parse all, each output from Dataset looks like 
            (x, word_emb, y) whereas we want (x, (word_emb, y)). So 
            the trick is to parse x and word_emb separately, and we 
            manually zip them into the right structure. 

            For actual usage, see `prepare_dataset`.
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
    if component == 'x':
        x = tf.io.parse_tensor(example['x'], out_type=float)
        x_length = [example['x_length']]
        x = tf.reshape(x, x_length)
        return x

    # semantic vector
    elif component == 'word_emb':
        word_emb = tf.io.parse_tensor(example['word_emb'], out_type=float)
        word_emb_length = [example['word_emb_length']]
        word_emb = tf.reshape(word_emb, word_emb_length)  # reshape, so shape becomes known.
        return word_emb

    # one hot label
    elif component == 'label':
        label = tf.io.parse_tensor(example['label'], out_type=tf.int64)
        label_length = [example['label_length']]
        label = tf.reshape(label, label_length)
        return label


if __name__ == '__main__':
    pass