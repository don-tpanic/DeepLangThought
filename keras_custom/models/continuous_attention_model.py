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

from keras_custom.layers.attention_model_layers import EmbedSemantics, \
     FineTuneOutput, ProduceAttentionWeights, ExpandAttention
from keras_custom.engine import training

"""
Attention models
"""

def vgg2_renamed():
    """
        To avoid circle in graph, all layers names must be unique,
        which means here we need to rename all vgg2 layers so keras does not
        confuse this with vgg1.
    """
    vgg2 = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

    vgg2_layers = []
    for layer in vgg2.layers:
        layer.trainable = False
        temp = layer.name
        layer._name = temp + '_branch'
        vgg2_layers.append(layer)
    # vgg2.summary()
    return vgg2, vgg2_layers


def continuous_model_v2(opt, num_categories=range(1000)):
    """
        trainable weights (total = 655872):
            1. vgg2 prediction -> embedding_layer (0)
            2. embedding_layer -> prior_attention_factory (768 * 512)
            3. prior_attention_factory -> attention_factory (512 * 512 + 512)
    """
    input = Input(shape=(224, 224, 3))

    # first we have the main branch before the attention layer
    WHERE_IS_ATTENTION = 'block4_pool'
    vgg1 = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

    # freeze all weights
    vgg1_layers = []
    for layer in vgg1.layers:
        layer.trainable = False
        vgg1_layers.append(layer)

    # cut off vgg1 at block4_pool
    cutoff = None
    for i, layer in enumerate(vgg1_layers):
        if layer.name == WHERE_IS_ATTENTION:
            cutoff = i
            break

    # stitch a new input layer until the block4_pool
    x = vgg1_layers[1](input)
    for layer in vgg1_layers[2:cutoff+1]:
        x = layer(x)

    num_c = int(x.shape[3])  ## for later use
    HWC = int(x.shape[1] * x.shape[2] * x.shape[3])
    x_mother = Flatten(name='flatten_mother')(x)
    print('CHECK: x_mother.shape = ', x_mother.shape)

    # --------------------------------------------------------------------------
    # right branch, where we now take in real input rather than pseudo input
    # REVIEW: it might be possible to use the same input from the main model but
    # REVIEW: the first attempt led to circle in the compute graph..
    vgg2, vgg2_layers = vgg2_renamed()
    x = vgg2_layers[1](input)
    for layer in vgg2_layers[2:]:
        x = layer(x)  # the last x here is the prediction out from vgg2

    # we connect the output from vgg2 to a semantic embedding layer
    x = EmbedSemantics(768, name='embedding_layer')(x)
    print(x.shape, 'before prior_attention_factory')
    # (None, 768)

    # a simple hidden FC layer, we may need more
    x = Dense(512, activation='relu', name='prior_attention_factory')(x)         # TEMP: change the size to very small number like 50
    print(x.shape, 'after prior_attention_factory')



    # TEMP (13/03/2020): use dropout to enforce stable weights for continuous(0)
    x = Dropout(0.1)(x)
    # print('# WARNING: dropout is OFF')
    # print('# WARNING: dropout is OFF')
    # print('# WARNING: dropout is OFF')
    # print('# WARNING: dropout is OFF')



    attention_weights = ProduceAttentionWeights(512, name='attention_factory')(x)
    # --------------------------------------------------------------------------
    # x_expanded is actually weights to be multiplied by x
    inShape = (None, 512)
    outdim = 100352
    x_expanded = ExpandAttention(outdim, input_shape=inShape, name='expanded_attention')(attention_weights)

    # --------------------------------------------------------------------------
    # on the main branch again.
    x_combined = Multiply()([x_mother, x_expanded])
    if WHERE_IS_ATTENTION == 'block4_pool':
        x = Reshape((14, 14, 512))(x_combined)
    for layer in vgg1_layers[cutoff+1:-1]:
        x = layer(x)  # x->(?, 4096)

    input_shape = x.shape
    outdim = len(num_categories)
    x = FineTuneOutput(outdim, input_shape=input_shape, name='fine_tune_output_1')(x)

    # --------------------------------------------------------------------------
    # compile to get the entire model
    model = training.Model_custom(inputs=input, outputs=x)
    model.compile(opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    # ==========================================================================
    # ---> need to use the orginal vgg16 prediction layer weights
    weights_and_biases_on_1k = vgg1.layers[-1].get_weights()
    ws = weights_and_biases_on_1k[0]
    bs = weights_and_biases_on_1k[1]
    sub_ws_bs = list([ws, bs])
    model.get_layer('fine_tune_output_1').set_weights(sub_ws_bs)

    # ---> for the expanded_attention layer, we need to sub the init weights to
    # an identity kernel
    transformation = np.tile(np.identity(512, dtype='float32'), [1, 196])
    model.get_layer('expanded_attention').set_weights([transformation])

    # ---> for the embedding layer, we sub the weights using bert semantics.
    emb_matrix = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')               # TODO: replace with np.ones() as control
    model.get_layer('embedding_layer').set_weights([emb_matrix])

    # plot_model(model, to_file='continuous_model_v2.png')
    # model.summary()
    return model


#
### A new model that is designed to be able to generalise across
### intensities and classes at the same time.
def continuous_model_master(opt, 
                            emb,
                            factory_width,
                            factory_depth,
                            activity_l1,
                            num_categories=range(1000)):
    """
    """
    ################## the attention() model part ##############################
    # first we have the main branch before the attention layer
    WHERE_IS_ATTENTION = 'block4_pool'
    vgg = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

    # freeze all weights
    vgg_layers = []
    for layer in vgg.layers:
        layer.trainable = False                                                 # comment out only for ALLFREE version
        vgg_layers.append(layer)

    # cut off vgg at block4_pool
    cutoff = None
    for i, layer in enumerate(vgg_layers):
        if layer.name == WHERE_IS_ATTENTION:
            cutoff = i
            break

    # stitch a new input layer until the block4_pool
    x = vgg_layers[cutoff].output

    num_c = int(x.shape[3])  ## for later use
    HWC = int(x.shape[1] * x.shape[2] * x.shape[3])
    x_mother = Flatten(name='flatten_mother')(x)

    ################## The attention factory network part ######################
    # input 2 - one-hot signal ready to go into BERT layer
    prior = Input(shape=(1000), name='input_prior')
    intensity = Input(shape=(1), name='input_intensity')
    # 768 or 30 depending on which embedding matrix we use
    if emb == 'BERT':
        emb_dim = 768
        layer_name = 'bert_embedding'
    elif emb == 'zprime':
        emb_dim = 30
        layer_name = 'unit_sphere'
    prior_embedded = EmbedSemantics(emb_dim, name=layer_name)(prior)
    x = Concatenate()([prior_embedded, intensity])
    print(x.shape)  # 769

    # -------------------- Into attention factory -------------------------- #
    for i in range(1, factory_depth+1):
        x = Dense(factory_width,
                  activation='relu',
                  name='hidden_layer_%s' % i,
                  kernel_initializer=keras.initializers.glorot_normal(),
                  activity_regularizer=keras.regularizers.l1(activity_l1)
                  )(x)
        # x = Dropout(0.1)(x)
        # skip v1: Add()
        # reshaped_intensity = np.ones(1024) * intensity
        # x = Add()([x, reshaped_intensity])
        # skip v2: concat(), units = 1025
        x = Concatenate()([x, intensity])
        #x = BatchNormalization()(x)   # BN has 4 params, 2 are trainable.

    print(x.shape)  # 4097 if factory_depth=1
    attention_weights = ProduceAttentionWeights(512,
                        name='produce_attention_weights')(x)
    print(attention_weights.shape)
    # --------------------- End of attention factory ------------------------- #
    
    ################## converge to the rest of the network #####################
    inShape = (None, 512)
    outdim = 100352
    x_expanded = ExpandAttention(outdim,
                                 input_shape=inShape,
                                 name='expanded_attention')(attention_weights)
    # on the main branch again.
    x_combined = Multiply()([x_mother, x_expanded])

    if WHERE_IS_ATTENTION == 'block4_pool':
        x = Reshape((14, 14, 512))(x_combined)
    for layer in vgg_layers[cutoff+1:-1]:
        x = layer(x)  # x->(?, 4096)

    input_shape = x.shape
    
    outdim = len(num_categories)
    x = FineTuneOutput(outdim,
                       input_shape=input_shape,
                       name='fine_tune_output_1')(x)
    # compile to get the entire model
    model = training.Model_custom(inputs=[vgg.input, prior, intensity], outputs=x)
    model.compile(opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    ### lastly, replacing some layer weights as needed.
    weights_and_biases_on_1k = vgg.layers[-1].get_weights()
    ws = weights_and_biases_on_1k[0]
    bs = weights_and_biases_on_1k[1]
    sub_ws_bs = list([ws, bs])
    model.get_layer('fine_tune_output_1').set_weights(sub_ws_bs)

    # ---> for the expanded_attention layer, we need to sub the init weights to
    # an identity kernel
    transformation = np.tile(np.identity(512, dtype='float32'), [1, 196])
    model.get_layer('expanded_attention').set_weights([transformation])

    # ---> for the embedding layer, we sub the weights using bert semantics.
    emb_dim = prior_embedded.shape[1]
    if emb_dim == 768:
        emb_matrix = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')     # TODO: replace with np.ones() as control
        model.get_layer('bert_embedding').set_weights([emb_matrix])
        
    elif emb_dim == 30:
        # Brett: non-negative embeddings
        emb_matrix = np.load('data_local/imagenet2vec/zprime_1k.npy')
        model.get_layer('unit_sphere').set_weights([emb_matrix])

    return model


# A sanity check model
def intersected_continuous_model_master(opt, 
                                        emb, 
                                        factory_width, 
                                        factory_depth, 
                                        activity_l1):
    """
    Intersect the continuous master model at the output
    of attention factory.
    This model has valid inputs (prior+intensity), the image input
    layer has no connection to any other layers so has no effect.
    The purpose of having this model is to see if the side network
    can learn the mapping from context pair to attention weights 
    when attention weights are used as labels from ICML results.
    """
    model = continuous_model_master(opt=opt, 
                                    emb=emb, 
                                    factory_width=factory_width,
                                    factory_depth=factory_depth,
                                    activity_l1=activity_l1)
    
    intersected_model = Model(inputs=model.input[1:], 
                              outputs=model.get_layer('produce_attention_weights').output)
    return intersected_model


if __name__ == '__main__':

    opt = Adam(lr=0.0003)
    opt = tensorflow.train.experimental.enable_mixed_precision_graph_rewrite(opt)

    #
    ###
    # model = continuous_model_v2(opt=opt)



    #
    ###
    # model = continuous_model_master(opt=opt, 
    #                                 emb='BERT', 
    #                                 factory_width=4096,
    #                                 factory_depth=4,
    #                                 activity_l1=1e-8)


    #
    ###
    model = intersected_continuous_model_master(
                                opt=opt, 
                                emb='BERT', 
                                factory_width=4096,
                                factory_depth=1,
                                activity_l1=1e-8)
    # plot_model(model, to_file='model_graph.png')
    model.summary()