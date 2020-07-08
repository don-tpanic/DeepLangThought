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

"""
The Language Models
"""

# temp: only has discrete labelling output
# TODO: deprecate once ALL-in-ONE model is complete
def lang_model_discrete(num_labels=1000, seed=42):
     """
     inputs:
     ------
          num_labels: label output layer size
     """
     vgg = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
     # load fine tuned weights:
     vgg.load_weights('_trained_weights/VGG16_finetuned_fullmodelWeights.h5')
     print('loaded in fine tuned VGG16 weights.')

     x = vgg.input
     # [1,-2] means skip input layer and stop at last FC.
     for layer in vgg.layers[1:-1]:
          layer.trainable = False
          x = layer(x)

     semantic_intermediate = Dense(768, activation='sigmoid', name='semantic_intermediate',
               kernel_initializer=keras.initializers.glorot_normal(seed=seed))(x)

     # attach discrete label layer.
     label_output = Dense(num_labels, activation='softmax', name='label_output',
               kernel_initializer=keras.initializers.glorot_normal(seed=seed))(semantic_intermediate)

     model = Model(inputs=vgg.input, outputs=label_output)

     # need to load in the trained weights for semantic layer from 
     # semantic only model
     trained_weights = np.load('_trained_weights/semantic_output_weights=29-06-20.npy',
                              allow_pickle=True)
     model.get_layer('semantic_intermediate').set_weights([trained_weights[0], trained_weights[1]])

     model.summary()
     return model


# temp: only has semantic output.
# TODO: deprecate once ALL-in-ONE model is complete
def lang_model_semantic(num_labels=1000, seed=42):
     """
     inputs:
     ------
          num_labels: label output layer size
     """
     vgg = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
     # load fine tuned weights:
     vgg.load_weights('_trained_weights/VGG16_finetuned_fullmodelWeights.h5')
     print('loaded in fine tuned VGG16 weights.')

     x = vgg.input
     # [1,-2] means skip input layer and stop at last FC.
     for layer in vgg.layers[1:-1]:
          layer.trainable = False
          x = layer(x)

     # attach the semantic layer (the end of the branch)
     semantic_output = Dense(768, activation='sigmoid', name='semantic_output',
               kernel_initializer=keras.initializers.glorot_normal(seed=seed))(x)

     model = Model(inputs=vgg.input, outputs=semantic_output)

     # TODO: free FC2
     model.get_layer('fc2').trainable = True


     model.summary()
     return model



def lang_model(num_labels=1000, seed=42):
     """
     One model has both semantic layer and discrete layer
     with the discrete layer being the output layer.

     The idea is that loss is composed of both semantic output 
     and discrete output.

     inputs:
     ------
          num_labels: label output layer size
     """
     vgg = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
     # load fine tuned weights:
     vgg.load_weights('_trained_weights/VGG16_finetuned_fullmodelWeights.h5')
     print('loaded in fine tuned VGG16 weights.')

     x = vgg.input
     # [1,-2] means skip input layer and stop at last FC.
     # we can change the last num to free up more layers.
     for layer in vgg.layers[1:-1]:
          layer.trainable = False
          x = layer(x)

     semantic_output = Dense(768, activation='sigmoid', name='semantic',
               kernel_initializer=keras.initializers.glorot_normal(seed=seed))(x)

     discrete_output = Dense(num_labels, activation='softmax', name='discrete',
               kernel_initializer=keras.initializers.glorot_normal(seed=seed))(semantic_output)

     model = Model(inputs=vgg.input, outputs=[semantic_output, discrete_output])
     model.compile(Adam(lr=3e-5),
                  loss=['mse', 'categorical_crossentropy'],
                  loss_weights=[1, 0],
                  metrics=['acc'],
                  )
     model.summary()
     #plot_model(model, to_file='lang_model.png')
     return model


if __name__ == '__main__':
     lang_model()

