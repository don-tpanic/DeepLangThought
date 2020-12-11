import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from keras_custom.generators.generator_wrappers import simple_generator
from EVAL.utils.data_utils import data_directory, load_classes

"""
Due to the pre-trained weights are suboptimal,
here we free up the entire VGG16 and fine tune it
to its absolute best and save the weights for 
future use.
"""


def execute(part='val_white'):
    # load model and fine tuned weights
    model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    model.load_weights('VGG16_finetuned_fullmodelWeights.h5')

    model.compile(Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['acc', 'sparse_top_k_categorical_accuracy'],
                  )

    per_lossW_acc = np.zeros((1000, 2))

    # test data
    directory = data_directory(part=part)
    wnids, indices, _ = load_classes(num_classes=1000, df='ranked')
    for i in range(len(wnids)):
        index = indices[i]
        wnid = wnids[i]
        print(f'*** currently eval index=[{index}] ***')
        gen, steps = simple_generator(
                            directory=directory,
                            classes=[wnid],
                            batch_size=16,
                            seed=42,
                            shuffle=True,
                            subset=None,
                            validation_split=0,
                            class_mode='sparse',  # need sparse to ensure label corrector has effect.
                            target_size=(224, 224),
                            preprocessing_function=preprocess_input,
                            horizontal_flip=False)

        loss, top1acc, top5acc = model.evaluate_generator(gen, steps, verbose=1, workers=3)
        per_lossW_acc[i, :] = [top1acc, top5acc]
    
    #np.save(f'RESULTS/{part}/accuracy/baseline.npy', per_lossW_acc)
    #print('per_lossW_acc saved.')