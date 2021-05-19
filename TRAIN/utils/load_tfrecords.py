import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np 
import tensorflow as tf
from functools import partial
from tensorflow.data.experimental import AUTOTUNE
from . import iterate_tfrecords 

"""
Data loading script using tfrecords.

We should be able to:
    1. Prepare Dataset object for training and testing
    2. For training, 
        - we need to create training/validation (10%)
        - needs to have shuffling
    3. Finegrain vs coarse labels need to be added correctly.
"""

def prepare_dataset(part='val_white', 
                    subset='training',
                    validation_split=0.1, 
                    batch_size=128,
                    sup=None):
    """
    Purpose:
    --------
        Once all simclr outputs are .tfrecords,
        Convert them into tf.Dataset for fitting.

        Here the preparation is general. In other words,
        it returns train/val/test set based on the user
        provided arguments.

    inputs:
    -------
        part: train / val_white
        subset: training / validation / None
        validation_split: provided via config
        batch_size: used to compute number of steps
        sup: one of the superordinates / None
    
    return:
    -------
        dataset: in the form (x, (semantic, labels))
        num_steps
    """
    # load data iterator
    dataset_iterator = iterate_tfrecords.DirectoryIterator(
                                directory=f'simclr_reprs/{part}',
                                classes=None,
                                subset=subset,
                                validation_split=validation_split,
                                sup=sup)
    # get all file paths
    filepaths = np.array(dataset_iterator._filepaths)
    # get corresponding labels (auto-inferred)
    labels = np.array(dataset_iterator.classes)

    # shuffle all (make sure file and label match)
    np.random.seed(999)
    ordered_indices = np.arange(len(filepaths), dtype=int)
    shuffled_indices = np.random.choice(
                                ordered_indices, 
                                size=len(ordered_indices), 
                                replace=False)
    filepaths = filepaths[shuffled_indices]
    labels = labels[shuffled_indices]
    # NOTE(ken), I checked filenames and labels match after shuffle.

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
    # so that input to model is now (x, (semantics, labels))
    # otherwise, won't work.
    dataset = tf.data.Dataset.zip((dataset_x, dataset_target))

    num_steps = np.ceil(len(labels) / batch_size)
    return dataset.batch(batch_size).prefetch(AUTOTUNE), num_steps


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
    # NOTE: 
    -------
        one trick we apply here is to have one extra argument
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
        'word_emb_length': tf.io.FixedLenFeature((), tf.int64)
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


if __name__ == '__main__':
    pass