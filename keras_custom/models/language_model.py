import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

# from keras_custom.layers.attention_model_layers import EmbedSemantics, \
#      FineTuneOutput, ProduceAttentionWeights, ExpandAttention

"""
The Language Model
"""

def lang_model(seed=42):
     vgg = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
     x = vgg.input
     # [1,-2] means skip input layer and stop at last FC.
     for layer in vgg.layers[1:-1]:
          layer.trainable = False
          x = layer(x)

     # attach the semantic layer (the end of the branch)
     semantic_output = Dense(768, activation='sigmoid', name='semantic_output',
               kernel_initializer=keras.initializers.glorot_normal(seed=seed))(x)

     semantic_intermediate = Dense(768, activation='sigmoid', name='semantic_intermediate',
               kernel_initializer=keras.initializers.glorot_normal(seed=seed))(x)

     # attach discrete label layer.
     label_output = Dense(1000, activation='softmax', name='label_output',
               kernel_initializer=keras.initializers.glorot_normal(seed=seed))(semantic_intermediate)

     model = Model(inputs=vgg.input, outputs=[semantic_output, label_output])

     #from tensorflow.keras.utils import plot_model
     #plot_model(model, to_file='lang_model.png')
     #model.summary()


if __name__ == '__main__':
     lang_model()


