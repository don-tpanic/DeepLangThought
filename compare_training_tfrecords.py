import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import numpy as np
import tensorflow as tf
from keras_custom.models.language_model import lang_model_contrastive
from TRAIN.utils.data_utils import load_config, specific_callbacks, data_directory
from TRAIN.utils.saving_utils import save_model_weights
from keras_custom.generators import load_tfrecords


# model = lang_model_contrastive(config)
# model.build(input_shape=(1, 2048))
directory = data_directory(part='val_white', tfrecords=True)
classes = None
batch_size = 1
validation_split = 0.1
lossW = 1
    
    
# for every lossW, we restart the timer.
start_time = time.time()
# model.compile(tf.keras.optimizers.Adam(lr=config['lr']),
#             loss=['mse', 'sparse_categorical_crossentropy'],
#             loss_weights=[1, lossW],
#             metrics=['acc'])

val_dataset, val_steps = load_tfrecords.prepare_dataset(
    directory=directory,
    classes=classes,
    subset='validation',
    validation_split=validation_split,
    batch_size=batch_size,
    sup=None)


# model.fit(train_dataset.repeat(config['epochs']),
#         epochs=config['epochs'], 
#         verbose=1, 
#         callbacks=[earlystopping, tensorboard],
#         validation_data=val_dataset.repeat(config['epochs']),
#         steps_per_epoch=train_steps,
#         validation_steps=val_steps,
#         max_queue_size=40, 
#         workers=3, 
#         use_multiprocessing=False)



