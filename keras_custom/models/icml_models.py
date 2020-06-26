import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= '3'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import h5py
#
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Layer, Flatten, Reshape, \
    Dense, Dropout, Input, Multiply, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.python.ops import gen_math_ops
from tensorflow.keras import backend as K

from keras_custom.engine import training
from keras_custom import constraints
from keras_custom.layers.attention_model_layers \
    import FineTuneOutput, ExpandAttention, AttLayer_Filter_at_branch

"""
Models (and variants) complied in ICML paper.
"""

def AttentionModel_FilterWise(num_categories, attention_mode, lr, opt=Adam(lr=0.0003)):
    WHERE_IS_ATTENTION = 'block4_pool'

    model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    layers = [l for l in model.layers]
    weights_and_biases_on_1k = model.layers[-1].get_weights()

    index = None
    for i, layer in enumerate(layers):
        if layer.name == WHERE_IS_ATTENTION:
            index = i
            break
    x = layers[index].output

    num_c = int(x.shape[3])  ## for later use
    HWC = int(x.shape[1] * x.shape[2] * x.shape[3])
    x_mother = Flatten(name='flatten_mother')(x)

    # above the same
    # --------------------------------------------------------------------------

    '''
    element-wise multiply shall happen here. alone with the other branch
    of the network with fixed (ones) input
    '''
    # the other branch:
    input_batch = Input(shape=(512,))

    # a layer takes a bunch of ones and multiply them by attention weights
    inShape = (None, 512)
    outdim = 512
    x_branch = AttLayer_Filter_at_branch(outdim, input_shape=inShape, name='att_layer_1')(input_batch)

    # some deterministic function expands layer output to match flattened layer
    # representation from the mother model (i.e. x = Flatten()(x))
    transformation = np.tile(np.identity(512, dtype='float32'), [1, 196])

    # --------------------------------------------------------------------------
    # x_expanded is actually weights to be multiplied by x
    inShape = (None, 512)
    outdim = 100352

    # HACK: mother branch output has a flattened shape, whereas here the attention branch the shape is in 512
    # HACK: which means, in order to merge the branches, one has to expand the attention branch output to the flattened shape
    # HACK: AND, most importantly, one has to make sure each of 512 value gets multiplied onto the same filter of the flattened output from the preceding layer of attention
    x_expanded = ExpandAttention(outdim, input_shape=inShape, name='expand_attention')(x_branch)

    branch_model = Model(inputs=input_batch, outputs=x_expanded)

    # --------------------------------------------------------------------------
    x_combined = Multiply()([x_mother, x_expanded])

    if WHERE_IS_ATTENTION == 'block4_pool':
        x = Reshape((14, 14, 512))(x_combined)
    for layer in layers[index+1:-1]:
        x = layer(x)  # x->(?, 4096)
    ############################################################################
    input_shape = x.shape
    outdim = len(num_categories)

    x = FineTuneOutput(outdim, input_shape=input_shape, name='fine_tune_output_1')(x)
    ############################################################################
    model = training.Model_custom(inputs=[model.input, branch_model.input], outputs=x)
    model.compile(
                  opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['acc', 'sparse_top_k_categorical_accuracy'],
                  )

    ws = weights_and_biases_on_1k[0][:, num_categories]
    bs = weights_and_biases_on_1k[1][num_categories]
    sub_ws_bs = list([ws, bs])

    model.get_layer('fine_tune_output_1').set_weights(sub_ws_bs)

    # change weights on dot_product  layer  as transformation
    model.get_layer('expand_attention').set_weights([transformation])

    # plot_model(model, to_file='filter_wise_attention.png')
    # model.summary()
    return model


def VGG_FT_ATT_MODEL_FILTER_v2(num_categories, attention_mode, lr, opt=Adam(lr=0.0003)):
    """
    Create options for training attention at a different layer.
    """
    WHERE_IS_ATTENTION = 'block3_pool'
    model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    layers = [l for l in model.layers]
    weights_and_biases_on_1k = model.layers[-1].get_weights()

    index = None
    for i, layer in enumerate(layers):
        if layer.name == WHERE_IS_ATTENTION:
            index = i
            break
    x = layers[index].output

    # NOTE: depending on `WHERE_IS_ATTENTION`
    # NOTE: if after block4_pool, 512, 14
    # NOTE: if after bock3_pool, 256, 28
    attention_filter_size = x.shape[-1]
    attention_spatial_size = x.shape[1]
    attention_unit_size = int(x.shape[1] * x.shape[2] * x.shape[3])
    x_mother = Flatten(name='flatten_mother')(x)
    # --------------------------------------------------------------------------

    '''
    element-wise multiply shall happen here. alone with the other branch
    of the network with fixed (ones) input
    '''
    # the other branch:
    input_batch = Input(shape=(attention_filter_size,))

    # a layer takes a bunch of ones and multiply them by attention weights
    inShape = (None, attention_filter_size)
    outdim = attention_filter_size
    x_branch = AttLayer_Filter_at_branch(outdim, input_shape=inShape, name='att_layer_1')(input_batch)

    # some deterministic function expands layer output to match flattened layer
    # representation from the mother model (i.e. x = Flatten()(x))
    transformation = np.tile(np.identity(attention_filter_size, dtype='float32'),
                             [1, attention_spatial_size * attention_spatial_size])

    # --------------------------------------------------------------------------
    # x_expanded is actually weights to be multiplied by x
    inShape = (None, attention_filter_size)
    outdim = attention_unit_size

    # HACK: mother branch output has a flattened shape, whereas here the attention branch the shape is in 512
    # HACK: which means, in order to merge the branches, one has to expand the attention branch output to the flattened shape
    # HACK: AND, most importantly, one has to make sure each of 512 value gets multiplied onto the same filter of the flattened output from the preceding layer of attention
    x_expanded = ExpandAttention(outdim, input_shape=inShape, name='dot_product')(x_branch)
    branch_model = Model(inputs=input_batch, outputs=x_expanded)
    # --------------------------------------------------------------------------
    x_combined = Multiply()([x_mother, x_expanded])
                                                                                # TODO: eventually need to remove hard coding
    if WHERE_IS_ATTENTION == 'block4_pool':
        x = Reshape((14, 14, 512))(x_combined)
    if WHERE_IS_ATTENTION == 'block3_pool':
        x = Reshape((28, 28, 256))(x_combined)

    for layer in layers[index+1:-1]:
        x = layer(x)  # x->(?, 4096)
    ############################################################################
    input_shape = x.shape
    outdim = len(num_categories)

    x = FineTuneOutput(outdim, input_shape=input_shape, name='fine_tune_output_1')(x)
    ############################################################################
    model = training.Model_custom(inputs=[model.input, branch_model.input], outputs=x)
    model.compile(opt, loss='sparse_categorical_crossentropy',
                  metrics=['acc'],  # sparse_top_k_categorical_accuracy
                  )

    ws = weights_and_biases_on_1k[0][:, num_categories]
    bs = weights_and_biases_on_1k[1][num_categories]
    sub_ws_bs = list([ws, bs])

    model.get_layer('fine_tune_output_1').set_weights(sub_ws_bs)

    # change weights on dot_product  layer  as transformation
    model.get_layer('dot_product').set_weights([transformation])

    # plot_model(model, to_file='filter_wise_attention.png')
    # model.summary()
    return model


if __name__ == '__main__':
    opt = Adam(lr=0.0003)
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

    # VGG_FT_ATT_MODEL_FILTER(num_categories=range(1000), attention_mode='BiVGG-FILTER', lr=0.0003, opt=opt)

    VGG_FT_ATT_MODEL_FILTER_v2(num_categories=range(1000), attention_mode='BiVGG-FILTER', lr=0.0003, opt=opt)
