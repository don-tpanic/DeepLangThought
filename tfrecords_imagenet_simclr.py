import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np 
import tensorflow as tf
from TRAIN.utils.data_utils import data_directory, load_classes
from keras_custom.generators import simclr_preprocessing

"""
Create tfrecords for all Imagenet (train & val_white)
associated simclr final outputs.
"""

def prepare_dataset(part='val_white'):
    
    test_dir = f'simclr_reprs/{part}/n01440764'

    # TEMP
    filepaths = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir)]
    # print(filepaths)
    dataset_x = tf.data.Dataset.from_tensor_slices(filepaths)

    def read_tfrecord(serialized_example):
        """

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


    dataset_x = dataset_x.interleave(tf.data.TFRecordDataset)
    return dataset_x.map(read_tfrecord)


def create_tfrecords():
    top_path = 'simclr_reprs/'
    model = load_model()
    wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')
    parts = ['val_white', 'train']

    for part in parts:
        directory = data_directory(part=part)
        wnids, labels, _ = load_classes(num_classes=1000, df='ranked')

        for wnid, label in zip(wnids, labels):
            create_single_tfrecord(model, 
                                   wordvec_mtx, 
                                   top_path, 
                                   directory, 
                                   part, 
                                   wnid, 
                                   label)
            exit()


def create_single_tfrecord(model, 
                           wordvec_mtx, 
                           top_path, 
                           directory, 
                           part, 
                           wnid, 
                           label):

    # create full path to save the record
    record_wnid_path = os.path.join(top_path, part, wnid)
    if not os.path.exists(record_wnid_path):
        os.makedirs(record_wnid_path)

    # full path to wnid
    class_path = os.path.join(directory, wnid)

    # all image names given wnid (NOT path)
    image_names = os.listdir(class_path)

    # create one hot label
    one_hot_label = np.zeros((1000)).astype(int)
    one_hot_label[label] = 1

    # create semantic label
    word_emb = np.dot(one_hot_label, wordvec_mtx).astype(np.float32)

    for image_name in image_names:

        # the full path to save a record
        record_image_path = os.path.join(record_wnid_path, f'{image_name[:-5]}.tfrecords')

        with tf.io.TFRecordWriter(record_image_path) as writer:

                # full path to load one image.
                image_path = os.path.join(class_path, image_name)

                # load, preprocess image and load into simclr.
                x = tf.keras.preprocessing.image.load_img(image_path)
                x = tf.keras.preprocessing.image.img_to_array(x)
                x = tf.convert_to_tensor(x, dtype=tf.uint8)
                x = simclr_preprocessing._preprocess(x, is_training=False)
                x = tf.reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]])
                x = model.predict(x)[0]  # otherwise len(x)=1

                # serialize and save this record.
                example = serialize_example(x=tf.io.serialize_tensor(x), 
                                            x_length=len(x), 
                                            word_emb=tf.io.serialize_tensor(word_emb), 
                                            word_emb_length=len(word_emb), 
                                            label=tf.io.serialize_tensor(one_hot_label),
                                            label_length=len(one_hot_label))
                writer.write(example)


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


def serialize_example(x, x_length, 
                      word_emb, word_emb_length, 
                      label, label_length):
    feature = {
        'x': _bytes_feature(x),
        'x_length': _int64_feature(x_length),
        'word_emb': _bytes_feature(word_emb),
        'word_emb_length': _int64_feature(word_emb_length),
        'label': _bytes_feature(label),
        'label_length': _int64_feature(label_length)
    }
    #  Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


if __name__ == '__main__':
    create_tfrecords()
    # dataset = prepare_dataset().batch(8)
    # for i in dataset:
    #     print(i)
    #     exit()