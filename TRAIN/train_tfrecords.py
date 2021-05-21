import time
import numpy as np
import tensorflow as tf
from keras_custom.models.language_model import lang_model_contrastive
from TRAIN.utils.data_utils import load_config, specific_callbacks, data_directory
from TRAIN.utils.saving_utils import save_model_weights
# from TRAIN.utils import load_tfrecords
from keras_custom.generators import load_tfrecords


def execute(config):
    model = lang_model_contrastive(config)
    # prev we didn't have to build, because now
    # we are headless.
    model.build(input_shape=(1, 2048))
    # lossWs = [1, 2, 3, 5, 7, 10, 0.1, 0]
    lossWs = [2, 3, 1]
    superordinates = [None]
    if 'finegrain' in config['config_version'] and len(superordinates) > 1:
        print(f'Error')
        exit()

    # TODO. should probably go into config.
    directory = data_directory(part='train', tfrecords=True)
    classes = None
    batch_size = config['batch_size']
    validation_split = config['validation_split']

    every_runtime = []
    for sup in superordinates:
        
        for lossW in lossWs:
            
            # for every lossW, we restart the timer.
            start_time = time.time()
            model.compile(tf.keras.optimizers.Adam(lr=config['lr']),
                        loss=['mse', 'sparse_categorical_crossentropy'],
                        loss_weights=[1, lossW],
                        metrics=['acc'])

            train_dataset, train_steps = load_tfrecords.prepare_dataset(
                        directory=directory,
                        classes=classes,
                        subset='training',
                        validation_split=validation_split,
                        batch_size=batch_size,
                        sup=sup)
            val_dataset, val_steps = load_tfrecords.prepare_dataset(
                        directory=directory,
                        classes=classes,
                        subset='validation',
                        validation_split=validation_split,
                        batch_size=batch_size,
                        sup=sup)

            if sup is not None:
                lossW = f'{lossW}-{sup}'
            earlystopping, tensorboard = specific_callbacks(config=config, lossW=lossW)

            # QUESTION: does val_dataset need repeat()?
            # QUESTION: .repeat(NUM_EPOCHS)?
            model.fit(train_dataset.repeat(config['epochs']),
                    epochs=config['epochs'], 
                    verbose=1, 
                    callbacks=[earlystopping, tensorboard],
                    validation_data=val_dataset.repeat(config['epochs']),
                    steps_per_epoch=train_steps,
                    validation_steps=val_steps,
                    max_queue_size=40, 
                    workers=3, 
                    use_multiprocessing=False)

            # save trained weights
            save_model_weights(model=model, config=config, lossW=lossW)

            # time it
            end_time = time.time()
            # in hrs.
            duration = (end_time - start_time) / 3600.
            every_runtime.append(duration)
            config_version = config['config_version']
            print(f'duration = {duration}')
            np.save(f'every_runtime_{config_version}.npy', every_runtime)



