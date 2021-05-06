import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yaml
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
#from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops import gen_math_ops
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Layer, Flatten, Reshape, Dense, \
     Dropout, Input, Multiply, Dot, Concatenate, BatchNormalization

"""
The Language Models
     1. lang_model: prior revision, uses VGG16 as perceptual-front-end
     2. lang_model_constrastive: uses simclr as perceptual-front-end
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
     vgg.load_weights('VGG16_finetuned_fullmodelWeights.h5')
     print('loaded in fine tuned VGG16 weights.')

     x = vgg.input
     # we can change the second index to
     # remove some layers such as the FC layers.

     # e.g. [1, :-1] --> keep FC1,2
     # e.g. [1, :-3] --> remove FC1,2
     for layer in vgg.layers[1:-1]:
          layer.trainable = False
          x = layer(x)



     # TODO: have brand new FC1,2 and randomly initialise them.
     #x = Dense(4096, )



     # add a number of Dense layer between FC2 and semantic
     for i in range(w2_depth):
          x = Dense(4096, activation='relu', name=f'w2_dense_{i}',
                    kernel_initializer=keras.initializers.glorot_normal(seed=seed))(x)
     
     # 4096 * 768 + 768 = 3146496
     semantic_output = Dense(768, activation=None, name='semantic',
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
     #plot_model(model, to_file='lang_model.png')
     return model


def lang_model_contrastive(config):
     """
     Purpose:
     -------- 
          This model uses pretrained simclr as front end.
          And follows with semantic layers, prediction layer as
          the same as the previous model.

     Impl:
     -----
          We use config to modify the setup of the model.
          When config is None, the model is the pretrained SIMCLR itself.
     """    
     class Model(tf.keras.Model):
          def __init__(self, config):
               """
               Load in the pretrained simclr
               And add the same layers as in previous version.
               """
               super(Model, self).__init__()
               self.config = config

               if self.config is not None:
                    self.saved_model = tf.saved_model.load(self.config['path'])

                    # TODO: hardcoded, better way to do this?
                    self.w2_dense0_layer = tf.keras.layers.Dense(4096, 
                                                                 activation='relu',
                                                                 name=f"w2_dense_0")
                    self.w2_dense1_layer = tf.keras.layers.Dense(4096, 
                                                                 activation='relu',
                                                                 name=f"w2_dense_1")
                    self.semantic_layer = tf.keras.layers.Dense(768, 
                                                            activation=None,
                                                            name='semantic_layer')
                    
                    # TODO: config['label_type'] == 'fine-grained, num_classes = 1000
                    self.classify_layer = tf.keras.layers.Dense(1000, 
                                                            activation='softmax',
                                                            name='discrete_layer')
                    # self.optimizer = LARSOptimizer(
                    #      learning_rate,
                    #      momentum=momentum,
                    #      weight_decay=weight_decay,
                    #      exclude_from_weight_decay=['batch_normalization', 'bias', 'head_supervised'])
               else: 
                    self.saved_model = tf.saved_model.load('r50_1x_sk0/saved_model/')
                    
          def call(self, inputs):
               """
               Straightforward feedforward net
               """
               simclr_outputs = self.saved_model(inputs, trainable=False)
               if self.config is None:
                    return simclr_outputs['final_avg_pool']
               else:
                    x = self.w2_dense0_layer(simclr_outputs['final_avg_pool'])
                    x = self.w2_dense1_layer(x)
                    semantic_output = self.semantic_layer(x)
                    classify_output = self.classify_layer(semantic_output)
                    return semantic_output, classify_output
     return Model(config)


if __name__ == '__main__':
     pass