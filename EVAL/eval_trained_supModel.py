import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

# from keras_custom.generators.generator_wrappers import lang_gen, simple_generator
from keras_custom.generators.generator_wrappers import data_generator_v2
from EVAL.utils.data_utils import data_directory, load_classes, load_config
from EVAL.utils.model_utils import ready_model_simclr

"""
To make sure the supGroup models trained 
do what we expect which is to only activate the supUnit 
not the others.

E.g. All 129 dogs will mostly only activate the first dog unit
at the output layer when lossW is high.
"""

def execute():

    config = load_config('simclr_coarsegrain_v2.1.run1')

    part = 'val_white'
    for lossW in ['1-reptile']:
        # Load entire trained model.
        model = ready_model_simclr(config, lossW, return_semantic=False)
        directory = data_directory(part=part)
        wnids, indices, categories = load_classes(num_classes=1000, df='reptile')


        percent_of151 = []


        # one class at a time,
        # each class, final result is an average 768 vector
        for i in range(len(wnids)):
            wnid = wnids[i]
            category = categories[i]
            index = indices[i]

            gen, steps = data_generator_v2(directory=directory,
                            classes=[wnid],
                            batch_size=config['batch_size'],
                            seed=config['generator_seed'],
                            shuffle=False,
                            subset=subset,
                            validation_split=config['validation_split'],
                            class_mode='categorical',
                            target_size=(224, 224),
                            preprocessing_function=preprocessing_function,
                            horizontal_flip=false, 
                            wordvec_mtx=None,
                            simclr_range=False,
                            simclr_augment=False,
                            sup='reptile')


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
        
        np.save(f'percent_of151-lossW={lossW}.npy', percent_of151)