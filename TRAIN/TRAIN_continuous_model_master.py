import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import time

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from keras_custom.generators.generator_wrapers import generator_for_batchMomentum
from keras_custom.callbacks import RelativeEarlyStopping, \
    ModelCheckpoint_and_save_attention

from keras_custom.models.continuous_attention_model import continuous_model_master
from TRAIN.utils.saving_utils import save_continuous_weights
from TRAIN.utils.data_utils import load_classes, data_directory


# Test
from keras_custom.callbacks import Checkpoint_AttnFactory
###


def custom_callbacks(run, patience, relative_improve_as_percent, factory_depth):
    """
    1. patience
    2. relative earlystopping
    3. save best weights not last epoch
    4. tensorBoard
    """
    NAME = 'masterCONTINUOUS_run{run}'.format(run=run)
    tensorboard = TensorBoard(log_dir='log/adv/{}'.format(NAME), 
                              update_freq='epoch')
    earlystopping = RelativeEarlyStopping(
                            monitor='val_loss',
                            min_perc_delta=relative_improve_as_percent/100,
                            patience=patience,
                            verbose=2,
                            mode='min',
                            restore_best_weights=False)

    # TODO: review.
    checkpoint = Checkpoint_AttnFactory(filepath='placeholder/', factory_depth=factory_depth, run=run)

    return tensorboard, earlystopping, checkpoint


def train_model(model, 
                run, patience, 
                imagenet_train, 
                batch_size, 
                epochs, 
                num_classes, 
                intensities, 
                relative_improve_as_percent,
                batchMomentum,
                factory_depth):

    wnids, indices, _ = load_classes(num_classes)

    # # if adversarial: wnids=1000, indices=200, TODO: this is a hack. improve.
    wnids = None 
    train_gen, train_steps = generator_for_batchMomentum(
                                directory=imagenet_train,
                                classes=wnids,
                                batch_size=batch_size,
                                seed=42,
                                shuffle=True,
                                subset='training',
                                validation_split=0.1,
                                class_mode='sparse',
                                target_size=(224, 224),
                                preprocessing_function=preprocess_input,
                                horizontal_flip=True,
                                # ----------------------
                                focus_indices=indices,
                                intensities=intensities,
                                batchMomentum=batchMomentum)

    val_gen, val_steps = generator_for_batchMomentum(
                                directory=imagenet_train,
                                classes=wnids,
                                batch_size=batch_size,
                                seed=42,
                                shuffle=True,
                                subset='validation',
                                validation_split=0.1,
                                class_mode='sparse',
                                target_size=(224, 224),
                                preprocessing_function=preprocess_input,
                                horizontal_flip=False,
                                # ----------------------
                                focus_indices=indices,
                                intensities=intensities,
                                batchMomentum=batchMomentum)
    # TODO: review.
    tensorboard, earlystopping, checkpoint = custom_callbacks(
                            run=run, 
                            patience=patience, 
                            relative_improve_as_percent=relative_improve_as_percent,
                            factory_depth=factory_depth)

    model.Master_fit_generator_custom(train_gen,
                                      steps_per_epoch=train_steps,
                                      epochs=epochs,
                                      validation_data=val_gen,
                                      validation_steps=val_steps,
                                      verbose=1,
                                      max_queue_size=40,
                                      workers=3,
                                      use_multiprocessing=False,
                                      callbacks=[tensorboard, earlystopping, checkpoint],
                                      num_classes=num_classes  # 25-5-2020: TODO, num_classes is now never used further. remove.
                                      )
    return model


def main(patience,
         lr,
         opt,
         emb,
         factory_width,
         factory_depth,
         imagenet_train,
         batch_size,
         epochs,
         num_classes,
         intensities,
         relative_improve_as_percent,
         activity_l1,
         batchMomentum,
         ):
    
    if len(intensities) == 1:
        Int = intensities[0]
    else:
        Int = ''
        for i in intensities:
            Int += '{i}+'.format(i=i)
        Int = Int[:-1]

    run_name = f'1epsilon=1-{num_classes}cls_v2prior_{emb}_{lr}_{batch_size}' \
               f'_WIDE={factory_width}_DEEP={factory_depth}' \
               f'-v2skip_Int={Int}' \
               f'-noDropout-BN-l1={activity_l1}-batchMomentum={batchMomentum}-patience={patience}'
        
    for run in [run_name]:
        model = continuous_model_master(opt=opt,
                                        emb=emb,
                                        factory_width=factory_width,
                                        factory_depth=factory_depth,
                                        activity_l1=activity_l1)

        model = train_model(model=model,
                            run=run,
                            patience=patience,
                            imagenet_train=imagenet_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            num_classes=num_classes,
                            intensities=intensities,
                            relative_improve_as_percent=relative_improve_as_percent,
                            batchMomentum=batchMomentum,
                            factory_depth=factory_depth)
        save_continuous_weights(model=model,
                                run=run,
                                factory_depth=factory_depth)

        K.clear_session()


def execute():
    start_time = time.time()
    imagenet_train = data_directory()
    patience = 30
    lr = 3e-5
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(Adam(lr=lr, epsilon=1))
    emb = 'BERT'
    factory_width = 4096
    factory_depth = 2
    activity_l1 = 0
    batch_size = 128
    epochs = 500  # default 500
    num_classes = 30
    intensities = [0.5]  # NOTE: in adv200, uniform != 1/num_classes but 1/1000
    relative_improve_as_percent = 0.   # prev 0.1%
    
    # how many consequtive batches to train on the same context pair.
    batchMomentum = 1  # default 1
    
    if True:
        main(patience=patience,
            lr=lr,
            opt=opt,
            emb=emb,
            factory_width=factory_width,
            factory_depth=factory_depth,
            imagenet_train=imagenet_train,
            batch_size=batch_size,
            epochs=epochs,
            num_classes=num_classes,
            intensities=intensities,
            relative_improve_as_percent=relative_improve_as_percent,
            activity_l1=activity_l1,
            batchMomentum=batchMomentum)

        dur = time.time() - start_time
        print('run time = {t} hrs'.format(t=dur/3600))