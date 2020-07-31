import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr, pearsonr

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras_custom.generators.generator_wrappers import lang_gen, simple_generator
from EVAL.utils.data_utils import data_directory, load_classes
from EVAL.utils.model_utils import ready_model

"""
To make sure the supGroup models trained 
do what we expect which is to only activate the supUnit 
not the others.

E.g. All 129 dogs will mostly only activate the first dog unit
at the output layer when lossW is high.
"""

def execute():
    ######################
    part = 'val_white'
    lr = 3e-5
    for lossW in ['0.1-sup=canidae', '1-sup=canidae', '10-sup=canidae']:
        version = '27-7-20'
        #discrete_frozen = False
        w2_depth = 2
        run_name = f'{version}-lr={str(lr)}-lossW={lossW}'
        intersect_layer = 'discrete'
        ######################


        model = ready_model(w2_depth=w2_depth, 
                            run_name=run_name, 
                            lossW=lossW, 
                            intersect_layer=intersect_layer)

        wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')
        directory = data_directory(part=part)
        wnids, indices, categories = load_classes(num_classes=1000, df='canidae')


        percent_of151 = []


        # one class at a time,
        # each class, final result is an average 768 vector
        for i in range(len(wnids)):
            wnid = wnids[i]
            category = categories[i]
            index = indices[i]

            gen, steps = simple_generator(
                            directory=directory,
                            classes=[wnid],
                            batch_size=128,
                            seed=42,
                            shuffle=True,
                            subset=None,
                            validation_split=0,
                            class_mode='sparse',  # only used for lang due to BERT indexing
                            target_size=(224, 224),
                            preprocessing_function=preprocess_input,
                            horizontal_flip=False)
            # [(, .768), (n, 1000)]
            outputs = model.predict_generator(gen, steps, verbose=0, workers=3)
            output_probs = outputs[1]
            # find the highest unit for each image given class.

            #highest_unit_per_class = []
            highest_unit_per_class = 0
            for i in range(output_probs.shape[0]):
                prob_i = output_probs[i, :]
                max_unit = np.argmax(prob_i)
                #highest_unit_per_class.append(max_unit)
                if max_unit == 151:
                    highest_unit_per_class += 1

            percent_of151_per_class = highest_unit_per_class / output_probs.shape[0]
            percent_of151.append(percent_of151_per_class)
            
            print(f'original class = [{index}]')
            print('percent of 151 per class = ', percent_of151_per_class)
            print('----------------------------')
        
        np.save(f'RESULTS/supCheck/percent_of151-lossW={lossW}.npy', percent_of151)