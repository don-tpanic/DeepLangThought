import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16

from keras_custom.models.language_model import lang_model

"""
Some pre-defined models that are used repeatedly.
"""

def ready_model(w2_depth, run_name, lossW):
    """
    Load in a specified model and intersect the activation after the 
    semantic layer.

    inputs:
    -------
        w2_depth: number of dense layer between FC2 and semantic layer
        run_name: f'{version}-lr={str(lr)}-lossW={lossW}'
        lossW: loss on the discrete term
    """
    # model
    model = lang_model(w2_depth=w2_depth)

    # load trained weights
    ## w2 dense weights
    for i in range(w2_depth):
        with open(f'_trained_weights/w2_dense_{i}-{run_name}.pkl', 'rb') as f:
            semantic_weights = pickle.load(f)

    ## semantic weights
    with open(f'_trained_weights/semantic_weights-{run_name}.pkl', 'rb') as f:
        semantic_weights = pickle.load(f)

    model.get_layer('semantic').set_weights([semantic_weights[0], semantic_weights[1]])
    model = Model(inputs=model.input, outputs=model.get_layer('semantic').output)

    model.compile(tf.keras.optimizers.Adam(lr=3e-5),
                  loss=['mse', 'categorical_crossentropy'],
                  loss_weights=[1, lossW],
                  metrics=['acc'])

    model.summary()
    print(f'run_name: {run_name}')
    return model


def vgg16_intersected_model(layer='block4_pool'):
    """
    intersected VGG16 at a given layer output
    """
    model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    intersected_output = model.get_layer(layer).output
    model = Model(inputs=model.input, outputs=intersected_output)

    model.compile(tf.keras.optimizers.Adam(lr=3e-5),
                  loss=['categorical_crossentropy'])

    print(f'loading VGG16 model until layer = {layer}')
    return model