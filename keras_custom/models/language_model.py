import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= '3'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops import gen_math_ops
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Layer, Flatten, Reshape, Dense, \
     Dropout, Input, Multiply, Dot, Concatenate, BatchNormalization

#from keras_custom.metrics import totalLoss

"""
The Language Models
"""


def lang_model(w2_depth, discrete_frozen=False, num_labels=1000, seed=42):
     """
     One model has both semantic layer and discrete layer
     with the discrete layer being the output layer.

     The idea is that loss is composed of both semantic output 
     and discrete output.

     inputs:
     ------
          trainable_discrete: if discrete layer is trainable or not.
          w2_depth: number of dense layers between FC2 and semantic layer.
          num_labels: label output layer size
     """
     vgg = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
     vgg.load_weights('_trained_weights/VGG16_finetuned_fullmodelWeights.h5')
     print('loaded in fine tuned VGG16 weights.')

     x = vgg.input
     # [1,-2] means skip input layer and FC2 is trainable.
     # we can change the last num to free up more layers.
     for layer in vgg.layers[1:-1]:
          layer.trainable = False
          x = layer(x)

     # add a number of Dense layer between FC2 and semantic
     for i in range(w2_depth):
          x = Dense(768, activation='relu', name=f'w2_dense_{i}',
                    kernel_initializer=keras.initializers.glorot_normal(seed=seed))(x)
     
     # 4096 * 768 + 768 = 3146496
     semantic_output = Dense(768, activation='sigmoid', name='semantic',
               kernel_initializer=keras.initializers.glorot_normal(seed=seed))(x)


     # 768 * 1000 + 1000 = 769000
     discrete_output = Dense(num_labels, activation='softmax', name='discrete',
               kernel_initializer=keras.initializers.glorot_normal(seed=seed))(semantic_output)
     model = Model(inputs=vgg.input, outputs=[semantic_output, discrete_output])

     # freeze w3
     if discrete_frozen:
          # 4096 * 768 + 768 = 3146496
          model.get_layer('discrete').trainable = False
          print('discrete layer is not trainable.')

     model.summary()
     plot_model(model, to_file='lang_model.png')
     return model


if __name__ == '__main__':
     lang_model()

