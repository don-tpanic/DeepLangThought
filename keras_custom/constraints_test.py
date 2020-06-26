import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import h5py

import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.models import Model
import constraints 
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Flatten, Reshape, Dense, \
     Input, Multiply, Dot


x_i = Input(shape=(12,))
x_o = Dense(10, kernel_constraint=constraints.Clip())(x_i)
model = Model(inputs=x_i, outputs=x_o)
model.summary()