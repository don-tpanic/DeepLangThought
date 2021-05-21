import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np 
import tensorflow as tf
from TRAIN.utils.data_utils import data_directory
from keras_custom.models.language_model import lang_model_contrastive
from keras_custom.generators import simclr_preprocessing

"""
1. Grab simclr output and save as tfrecords.
2. Add BERT-labels(768) into tfrecords.
"""

def load_model():
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

model = load_model()       
wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')


train_path = data_directory(part='train')
class_path = f'{train_path}/n02168699'


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image, label, image_shape, word_emb):
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'word_emb': _bytes_feature(word_emb)
    }
    #  Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def read_tfrecord(serialized_example):
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'depth': tf.io.FixedLenFeature((), tf.int64),
        'word_emb': tf.io.FixedLenFeature((), tf.string)
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    image = tf.io.parse_tensor(example['image'], out_type=float)
    image_shape = [example['height'], example['width'], example['depth']]
    image = tf.reshape(image, image_shape)
    word_emb = tf.io.parse_tensor(example['word_emb'], out_type=float)

    return image, word_emb, example['label']




# tfrecord_dataset = tf.data.TFRecordDataset('simclr_reprs/n02168699.tfrecords')
# parsed_dataset = tfrecord_dataset.map(read_tfrecord)

# for i, data in enumerate(parsed_dataset.take(1)):
#     img = data[0]
#     print(img.shape)
#     word_emb = data[1]
#     print(word_emb.shape)