import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from TRAIN.utils.data_utils import load_classes, data_directory
from keras_custom.generators.generator_wrappers import lang_gen, simclr_gen


"""
Visualize the image output from generators with 
    1. Original preprocessed output 
    2. Simclr preprocessed output
""" 

def compare_generator_outputs(generator_list=['lang', 'simclr']):
    fig, ax = plt.subplots(1, 2)

    for i in range(len(generator_list)):
        gen_name = generator_list[i]
        if gen_name == 'lang':
            generator = lang_gen
            preprocessing_function = preprocess_input
        else:
            generator = simclr_gen
            preprocessing_function = None
        gen, steps = generator(
            data_directory(part='train'),
            classes=None,
            batch_size=1,
            seed=42,
            shuffle=False,
            subset='validation',
            validation_split=0.001,
            class_mode='categorical',
            target_size=(224, 224),
            preprocessing_function=preprocessing_function,
            horizontal_flip=False, 
            wordvec_mtx=np.load('data_local/imagenet2vec/imagenet2vec_1k.npy'))

        x, y = gen.__getitem__(idx=0)
        img = x[0]
        wordvec = y[0]
        label = y[1]
        print(wordvec.shape)
        ax[i].imshow(img)
        ax[i].set_title(f'{gen_name}')

    plt.savefig('check_generator_output.png')


def check_simclr_gen_outputs(num_imgs=4):
    fig, ax = plt.subplots(1, num_imgs)

    gen, steps = simclr_gen(
        data_directory(part='train'),
        classes=None,
        batch_size=num_imgs,
        seed=42,
        shuffle=True,
        subset='validation',
        validation_split=0.1,
        class_mode='categorical',
        target_size=(224, 224),
        preprocessing_function=None,
        horizontal_flip=False, 
        wordvec_mtx=np.load('data_local/imagenet2vec/imagenet2vec_1k.npy'),
        simclr_augment=False)

    x, y = gen.__getitem__(idx=0)

    for i in range(x.shape[0]):
        img = x[i]
        ax[i].imshow(img)

    plt.savefig('simclr_generator_outputs2.png')


if __name__ == '__main__':
    # compare_generator_outputs()
    check_simclr_gen_outputs()
