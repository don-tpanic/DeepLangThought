import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras_custom.generators.generator_wrappers import lang_gen, simple_generator
from EVAL.utils.data_utils import data_directory, load_classes
from EVAL.utils.model_utils import vgg16_intersected_model


def layer_activation(part, layer):
    model = vgg16_intersected_model(layer=layer)
    directory = data_directory(part=part)
    wnids, indices, categories = load_classes(num_classes=1000, df='ranked')

    # one class at a time,
    # each class, final result is an average 768 vector
    for i in range(len(wnids)):
        wnid = wnids[i]
        category = categories[i]

        gen, steps = simple_generator(
                        directory=directory,
                        classes=[wnid],
                        batch_size=128,
                        seed=42,
                        shuffle=True,
                        subset=None,
                        validation_split=0,
                        class_mode='sparse',
                        target_size=(224, 224),
                        preprocessing_function=preprocess_input,
                        horizontal_flip=False)
        # (N, 14, 14, 512)
        activations = model.predict(gen, steps, verbose=1, workers=3)
        # (512,)
        activations = np.mean(activations, axis=(0, 1, 2))
        assert activations.shape == (model.output.shape[-1], ), "average over image and spatial wrong"

        np.save(f'_computed_activations/{layer}=vgg16/{category}.npy', activations)




def execute():
    ##########
    part = 'val_white'
    layer = 'block4_pool'
    ##########

    #layer_activation(part=part, layer=layer)

