import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= '3'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


     w2_dense_0 (Dense)           (None, 4096)              16781312  
     _________________________________________________________________
     w2_dense_1 (Dense)           (None, 4096)              16781312  
     _________________________________________________________________
     semantic (Dense)             (None, 768)               3146496   
     _________________________________________________________________
     discrete (Dense)             (None, 1000)              769000    
     =================================================================
     Total params: 171,738,664
     Trainable params: 37,478,120
     Non-trainable params: 134,260,544
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


def lang_model_contrastive(config, return_semantic=False):
     """
     Purpose:
     -------- 
          This model uses pretrained vgg16 or simclr as front end.
          And follows with semantic layers, prediction layer as
          the same as the previous model.

     Impl:
     -----
          We use config to modify the setup of the model.
          When config is None, the model is the pretrained SIMCLR itself.

     if front_end == 'simclr':
          Layer (type)                 Output Shape              Param #   
          =================================================================
          w2_dense_0 (Dense)           multiple                  8392704   
          _________________________________________________________________
          w2_dense_1 (Dense)           multiple                  16781312  
          _________________________________________________________________
          semantic_layer (Dense)       multiple                  3146496   
          _________________________________________________________________
          discrete_layer (Dense)       multiple                  769000    
          =================================================================
          Total params: 29,089,512
          Trainable params: 29,089,512
          Non-trainable params: 0
     
     elif front_end == 'vgg16':
          w2_dense_0 (Dense)           (None, 4096)              16781312  
          _________________________________________________________________
          w2_dense_1 (Dense)           (None, 4096)              16781312  
          _________________________________________________________________
          semantic (Dense)             (None, 768)               3146496   
          _________________________________________________________________
          discrete (Dense)             (None, 1000)              769000    
          =================================================================
          Total params: 171,738,664
          Trainable params: 37,478,120
          Non-trainable params: 134,260,544
     """    
     if config['mixed_precision'] is True:
          tf.keras.mixed_precision.set_global_policy('mixed_float16')
          print('[Check] Setting global policy to mixed_float16..')

     if config['front_end'] == 'simclr':
          seed = config['kernel_seed']
          if config['kernel_initializer'] == 'glorot_normal':
               kernel_initializer = tf.keras.initializers.glorot_normal(seed=seed)
          # prior v2.2.run1 (we used glorot uniform with no seed, which is tf default)
          else:
               kernel_initializer = tf.keras.initializers.glorot_uniform()
          print(f'[Check] kernel initializer = ', config['kernel_initializer'])

          class LangModel(tf.keras.Model):
               def __init__(self, config, return_semantic):
                    """
                    Load in the pretrained simclr
                    And add the same layers as in previous version.
                    """
                    super(LangModel, self).__init__()
                    self.config = config
                    self.return_semantic = return_semantic

                    # --- when not headless, load the simclr front end --- #
                    if self.config['headless'] is False:
                         self.saved_model = tf.saved_model.load(self.config['path'])

                    # --- define layers that are used regardless of headless or not. --- #
                    self.w2_dense0_layer = tf.keras.layers.Dense(4096, 
                                                                 activation='relu',
                                                                 name=f"w2_dense_0",
                                                                 kernel_initializer=kernel_initializer)
                    self.w2_dense1_layer = tf.keras.layers.Dense(4096, 
                                                                 activation='relu',
                                                                 name=f"w2_dense_1",
                                                                 kernel_initializer=kernel_initializer)
                    self.semantic_layer = tf.keras.layers.Dense(768, 
                                                                activation=None,
                                                                name='semantic_layer',
                                                                kernel_initializer=kernel_initializer)  

                    # --- define discrete layer or not depending on whether to return semantic only. --- #
                    if self.return_semantic is False:                
                         if config['mixed_precision'] is False:          
                              self.classify_layer = tf.keras.layers.Dense(1000, 
                                                                      activation='softmax',
                                                                      name='discrete_layer',
                                                                      kernel_initializer=kernel_initializer)
                         # NOTE(ken), when use mixed precision 
                         # based on: https://www.tensorflow.org/guide/mixed_precision
                         else:
                              # default activation to Dense of `None`.
                              self.classify_layer = tf.keras.layers.Dense(1000,
                                                                      activation=None,
                                                                      name='discrete_layer',
                                                                      kernel_initializer=kernel_initializer)

                              print(f'[Check] classify_layer dtype = {self.classify_layer.dtype_policy}')
                              self.softmax_layer = tf.keras.layers.Activation('softmax', dtype='float32', name='softmax')
                              print(f'[Check] softmax_layer dtype = {self.softmax_layer.dtype_policy}')

               def call(self, inputs):
                    """
                    Straightforward feedforward net
                    """
                    # --- if not headless --- #
                    if self.config['headless'] is False:
                         inputs = self.saved_model(inputs, trainable=False)['final_avg_pool']
                    
                    # --- regardless of headless, the rest is the same --- #
                    x = self.w2_dense0_layer(inputs)
                    x = self.w2_dense1_layer(x)
                    semantic_output = self.semantic_layer(x)

                    # We only return semantic output
                    # when eval trained models
                    if return_semantic is True:
                         return semantic_output
                    # When training full model,  we include the classifier.
                    else:
                         if config['mixed_precision'] is False:
                              classify_output = self.classify_layer(semantic_output)
                         else:
                              # due to mix precision, we do in two steps.
                              logits = self.classify_layer(semantic_output)
                              print(f'[Check] logits.dtype = {logits.dtype.name}')
                              classify_output = self.softmax_layer(logits)
                              print(f'[Check] softmax.dtype = {classify_output.dtype.name}')
                         return semantic_output, classify_output
          return LangModel(config, return_semantic)


     elif config['front_end'] == 'vgg16':

          w2_depth = config['w2_depth']
          seed = 42

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

          # add a number of Dense layer between FC2 and semantic
          for i in range(w2_depth):
               x = Dense(4096, activation='relu', name=f'w2_dense_{i}',
                         kernel_initializer=keras.initializers.glorot_normal(seed=seed))(x)
          
          # 4096 * 768 + 768 = 3146496
          semantic_output = Dense(768, activation=None, name='semantic_layer',
                    kernel_initializer=keras.initializers.glorot_normal(seed=seed))(x)

          # if return semantics, we intercept at semantic layer
          if return_semantic is True:
               model = Model(inputs=vgg.input, outputs=semantic_output)
          # if False, we have two outputs.
          else:
               # 768 * 1000 + 1000 = 769000
               discrete_output = Dense(1000, activation='softmax', name='discrete_layer',
                         kernel_initializer=keras.initializers.glorot_normal(seed=seed))(semantic_output)
               model = Model(inputs=vgg.input, outputs=[semantic_output, discrete_output])
          model.summary()
          #plot_model(model, to_file='lang_model.png')
          return model


if __name__ == '__main__':
     pass