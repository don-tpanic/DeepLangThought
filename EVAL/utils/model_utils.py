import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16

from keras_custom.models.language_model import lang_model

"""
Some pre-defined models that are used repeatedly.
"""

def ready_model(version, lossW):
    """
    Load in a specified model and intersect the activation after the 
    semantic layer (either output or intermediate)
    """
    # model
    model = lang_model()
    # weights look like [kernel(4096,768), bias(768,)]
    trained_weights = np.load(f'_trained_weights/semantic_weights-{version}-lr=3e-05-lossW={lossW}.npy',
                            allow_pickle=True)

    model.get_layer('semantic').set_weights([trained_weights[0], trained_weights[1]])
    model = Model(inputs=model.input, outputs=model.get_layer('semantic').output)
    model.compile(tf.keras.optimizers.Adam(lr=3e-5),
                  loss=['mse', 'categorical_crossentropy'],
                  loss_weights=[1, lossW],
                  metrics=['acc'])


    print(f'version=[{version}], lossW=[{lossW}]')
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