import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import yaml

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras_custom.models.language_model import lang_model_contrastive
from keras_custom.preprocessing.image_dataset import image_dataset_from_directory
from keras_custom.generators.generator_wrappers import lang_gen, simclr_gen

from TRAIN.utils.data_utils import load_classes, data_directory
from keras_custom.generators.simclr_preprocessing import _preprocess


"""
Compare how the semantic representations 
from VGG16(fc2) and SIMCLRv2 differ.

We use validation set to save compute.
"""

# random_img = tf.random.uniform((1, 224,224,3))
# print(random_img)
# x = _preprocess(random_img)
# print(x)
# exit()


# model = lang_model_contrastive(config=None)
# model.compile(tf.keras.optimizers.Adam(),
#              loss=['mse'])

gen, steps = simclr_gen(
                    data_directory(part='train'),
                    classes=None,
                    batch_size=128,
                    seed=42,
                    shuffle=True,
                    subset='validation',
                    validation_split=0.1,
                    class_mode='categorical',
                    target_size=(224, 224),
                    preprocessing_function=None,
                    horizontal_flip=False, 
                    wordvec_mtx=np.load('data_local/imagenet2vec/imagenet2vec_1k.npy'))
print(next(gen))
exit()
# [(bs, 224,224,3)], [(bs, 1000), (bs, 768)]]

train_ds = image_dataset_from_directory(
                data_directory(part='train'),
                validation_split=0.1,
                subset="training",
                seed=42,
                image_size=(224, 224),
                batch_size=1,
                label_mode='int',
                wordvec_mtx=np.load('data_local/imagenet2vec/imagenet2vec_1k.npy'))

    
# def map_func(x, y):
    # return _preprocess(x), y

# train_ds = train_ds.map(lambda x, y: _preprocess(x))
# for x, y in train_ds:
#     print(x, y)
#     # simclr_out = model.predict(x)
#     # print(simclr_out.shape)  # (1, 2048)
#     exit()

# gen = lambda: (i for i in gen)
# dataset = tf.data.Dataset.from_generator(gen, output_signature=(
#          tf.RaggedTensorSpec(shape=(None, 1000), dtype=tf.int32),
#          tf.RaggedTensorSpec(shape=(None, 768), dtype=tf.float32)))

# ds = dataset.map(_preprocess).batch(1)