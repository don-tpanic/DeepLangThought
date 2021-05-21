import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import numpy as np
import tensorflow as tf
from keras_custom.generators.generator_wrappers import data_generator_v2
import matplotlib.pyplot as plt


gen, steps = data_generator_v2(directory=f'/mnt/fast-data17/datasets/ILSVRC/2012/clsloc/val_white',
                        classes=None,
                        batch_size=1,
                        seed=42,
                        shuffle=True,
                        subset='validation',
                        validation_split=0.1,
                        class_mode='sparse',
                        target_size=(224, 224),
                        preprocessing_function=None,
                        horizontal_flip=False, 
                        wordvec_mtx=None,
                        simclr_range=True,
                        simclr_augment=False,
                        sup=None)

out = gen.__getitem__(idx=5)
print(out)
img = out[0]
plt.imshow(img[0])
plt.savefig('img.png')



