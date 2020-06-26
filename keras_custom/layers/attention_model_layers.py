import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= '3'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import gen_math_ops
from tensorflow.keras import backend as K
from keras_custom import constraints

"""
Custom layers that are used in building attention models.
"""

# master continuous model
class EmbedSemantics(Layer):
    """
    Upon receiving (None, 1000) probabilities, convert to
    a predefined (weights later loaded) embeddinig space.
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(EmbedSemantics, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=keras.initializers.Ones(),
                                      trainable=False)

        super(EmbedSemantics, self).build(input_shape)

    def call(self, x):
        # x in (?,1000)
        # in order to broadcast,
        # we must have (?,1000,1) * (1000,768)
        x = K.expand_dims(x, axis=-1)
        # should be (?, 1000, 768)
        x = gen_math_ops.mul(x, self.kernel)
        # should be (?, 768)
        x = K.sum(x, axis=1)
        x = keras.activations.sigmoid(x)
        # NOTE: I confirm all values [0,1]
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = super(EmbedSemantics, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config


# master continuous model & ICML
class FineTuneOutput(Layer):
    """
    input:
    ------
        dim = (?, 4096)
    output:
    -------
        dim = (?, len(num_categories))
    replacement for the previous `prediction layer (dim=1000)`,
    weights_biases init: grab the w and b from `prediction layer` and only
    keep the connections from the `fc2 Dense` to num_categories units.
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FineTuneOutput, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=keras.initializers.Ones(),
                                      trainable=False)

        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim),
                                    initializer='zeros',
                                    trainable=False)
        super(FineTuneOutput, self).build(input_shape)

    def call(self, x):
        output = K.dot(x, self.kernel)
        output = K.bias_add(output, self.bias, data_format='channels_last')
        return keras.activations.softmax(output, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = super(FineTuneOutput, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config


# master continuous model & ICML
class ProduceAttentionWeights(Layer):
    """
    given some input, and output 512 activation vector which is used
    as attention weights.
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ProduceAttentionWeights, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=keras.initializers.glorot_normal(),
                                      trainable=True)

        super(ProduceAttentionWeights, self).build(input_shape)

    def call(self, x):
        output = K.dot(x, self.kernel)
        output = keras.activations.relu(output)
        output = K.clip(output, min_value=K.epsilon(), max_value=None)
        # NOTE: checked, output >= epsilon
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = super(ProduceAttentionWeights, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config


# master continuous model & ICML
class ExpandAttention(Layer):
    """
    Taking the 512 attention weights, expand to 100352 so that
    the flattened mother representation can be multiplied with the outputs
    of this layer, which results in an filter wise attention tuning.
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ExpandAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=keras.initializers.Ones(),
                                      trainable=False)
        super(ExpandAttention, self).build(input_shape)

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = super(ExpandAttention, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config


# ICML attention layer
class AttLayer_Filter_at_branch(Layer):
    """
    which is the same as `AttLayer` with no reg
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(AttLayer_Filter_at_branch, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1],),
                                      initializer=keras.initializers.Ones(),
                                      # regularizer=KO.l1_c(1e-3, c=1),
                                      constraint=constraints.Clip(),
                                      trainable=True)
        super(AttLayer_Filter_at_branch, self).build(input_shape)

    def call(self, x):
        return gen_math_ops.mul(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)