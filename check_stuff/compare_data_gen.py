import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras_custom.models.language_model import lang_model_contrastive
from keras_custom.generators.generator_wrappers import data_generator_v2, lang_gen
from TRAIN.utils.data_utils import load_config, specific_callbacks, data_directory


# gen, steps = data_generator_v2(
#                 directory=data_directory(part='val_white'),
#                 classes=None,
#                 batch_size=1,
#                 seed=42,
#                 shuffle=True,
#                 subset='validation',
#                 validation_split=0.1,
#                 class_mode='categorical',
#                 target_size=(224, 224),
#                 preprocessing_function=preprocess_input,
#                 horizontal_flip=False, 
#                 wordvec_mtx=np.load('data_local/imagenet2vec/imagenet2vec_1k.npy'),
#                 simclr_range=False,
#                 simclr_augment=False,
#                 sup=None)



gen, steps = lang_gen(
                directory=data_directory(part='val_white'),
                classes=None,
                batch_size=1,
                seed=42,
                shuffle=True,
                subset='validation',
                validation_split=0.1,
                class_mode='categorical',  # only used for lang due to BERT indexing
                target_size=(224, 224),
                preprocessing_function=preprocess_input,
                horizontal_flip=False, 
                wordvec_mtx=np.load('data_local/imagenet2vec/imagenet2vec_1k.npy'))

outs = gen.__getitem__(0)
print((outs[0]))