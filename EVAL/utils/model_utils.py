import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model

from keras_custom.models.language_model import lang_model_semantic, lang_model_discrete

"""
Some pre-defined models that are used repeatedly.
"""

def ready_model(model_type, version):
    """
    Load in a specified model and intersect the activation after the 
    semantic layer (either output or intermediate)
    """
    # model
    if model_type == 'semantic':
        model = lang_model_semantic()
        # weights look like [kernel(4096,768), bias(768,)]
        trained_weights = np.load(f'_trained_weights/semantic_output_weights={version}.npy',
                                allow_pickle=True)
        model.get_layer('semantic_output').set_weights([trained_weights[0], trained_weights[1]])
        model.compile(tf.keras.optimizers.Adam(lr=3e-5),
                  loss=[tf.keras.losses.MSE],
                  metrics='mean_squared_error',
                  )

    elif model_type == 'discrete':
        model = lang_model_discrete()
        trained_weights = np.load(f'_trained_weights/semantic_intermediate_weights={version}.npy',
                                allow_pickle=True)
        model.get_layer('semantic_intermediate').set_weights([trained_weights[0], trained_weights[1]])

        model = Model(inputs=model.input, outputs=model.get_layer('semantic_intermediate').output)

        model.compile(tf.keras.optimizers.Adam(lr=3e-5),
                  loss='categorical_crossentropy',
                  metrics=['acc', 'top_k_categorical_accuracy'],
                  )
    print(f'compiling [{model_type}] model, version [{version}] ...')
    return model