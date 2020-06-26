import os
import warnings
import numpy as np
import pandas as pd

from tensorflow.keras import backend as K
from tensorflow.keras import Model

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import monitoring

from tensorflow.python.keras.engine import training_utils  #, training_v2_utils, 
from keras_custom.engine import training_generator, Master_training_generator

# TODO: failed to pass some monitoring checks.
# _keras_api_gauge = monitoring.BoolGauge('/tensorflow/api/keras',
#                                         'keras api usage', 'method')

class Model_custom(Model):


    # TODO (29-05-2020): change to use fit
    ### Very messy if use fit due to native tf does not
    ### have `class_weight` for `test_on_batch()`

    def fit_generator_custom(self,
                              generator,
                              steps_per_epoch=None,
                              epochs=1,
                              verbose=1,
                              callbacks=None,
                              validation_data=None,
                              validation_steps=None,
                              validation_freq=1,
                              train_class_weight=None,
                              val_class_weight=None,
                              max_queue_size=10,
                              workers=1,
                              use_multiprocessing=False,
                              shuffle=True,
                              initial_epoch=0):
      # TODO: failed to pass some monitoring checks.
      # _keras_api_gauge.get_cell('fit_generator_custom').set(True)
      return training_generator.fit_generator(
                              self,
                              generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              verbose=verbose,
                              callbacks=callbacks,
                              validation_data=validation_data,
                              validation_steps=validation_steps,
                              validation_freq=validation_freq,
                              train_class_weight=train_class_weight,
                              val_class_weight=val_class_weight,
                              max_queue_size=max_queue_size,
                              workers=workers,
                              use_multiprocessing=use_multiprocessing,
                              shuffle=shuffle,
                              initial_epoch=initial_epoch,
                              steps_name='steps_per_epoch')

    def train_on_batch_custom(self, 
                              x, 
                              y=None,
                              sample_weight=None,
                              class_weight=None,
                              reset_metrics=True,
                              return_dict=False):
        
        self._assert_compile_was_called()
        self._check_call_args('train_on_batch')
        with self.distribute_strategy.scope(), \
         training_utils.RespectCompiledTrainableState(self):
         iterator = data_adapter.single_batch_iterator(
                            self.distribute_strategy, 
                            x,
                            y, 
                            sample_weight,
                            class_weight)
         train_function = self.make_train_function()
         logs = train_function(iterator)

        if reset_metrics:
          self.reset_metrics()
        logs = tf_utils.to_numpy_or_python_type(logs)
        if return_dict:
          return logs
        else:
          results = [logs.get(name, None) for name in self.metrics_names]
          if len(results) == 1:
            return results[0]
          return results

    def test_on_batch_custom(self, 
                             x, 
                             y=None, 
                             class_weight=None, 
                             sample_weight=None, 
                             reset_metrics=True,
                             return_dict=False):
        self._assert_compile_was_called()
        self._check_call_args('test_on_batch')
        with self.distribute_strategy.scope():
          iterator = data_adapter.single_batch_iterator(
                            self.distribute_strategy, 
                            x,
                            y, 
                            sample_weight,
                            class_weight)
          test_function = self.make_test_function()
          logs = test_function(iterator)
        
        if reset_metrics:
          self.reset_metrics()
        logs = tf_utils.to_numpy_or_python_type(logs)
        if return_dict:
            return logs
        else:
          results = [logs.get(name, None) for name in self.metrics_names]
        if len(results) == 1:
          return results[0]
        return results

    # TODO: learn from ICML upgrade and update the code to be the same tf2.2 compatiple.

    def Master_fit_generator_custom(self,
                                    generator,
                                    steps_per_epoch=None,
                                    epochs=1,
                                    verbose=1,
                                    callbacks=None,
                                    validation_data=None,
                                    validation_steps=None,
                                    validation_freq=1,
                                    max_queue_size=10,
                                    workers=1,
                                    use_multiprocessing=False,
                                    shuffle=True,
                                    initial_epoch=0,
                                    train_class_weight=None,
                                    val_class_weight=None,
                                    num_classes=None):

        return Master_training_generator.fit_generator(
                                        self,
                                        generator,
                                        steps_per_epoch=steps_per_epoch,
                                        epochs=epochs,
                                        verbose=verbose,
                                        callbacks=callbacks,
                                        validation_data=validation_data,
                                        validation_steps=validation_steps,
                                        validation_freq=validation_freq,
                                        max_queue_size=max_queue_size,
                                        workers=workers,
                                        use_multiprocessing=use_multiprocessing,
                                        shuffle=shuffle,
                                        initial_epoch=initial_epoch,
                                        steps_name='steps_per_epoch',
                                        train_class_weight=None,
                                        val_class_weight=None,
                                        num_classes=None)

    def Master_train_on_batch_custom(self, 
                                     x, 
                                     y,
                                     sample_weight=None,
                                     class_weight=None,
                                     reset_metrics=True,
                                     num_classes=None,
                                     return_dict=False):
        """
        Difference to `train_on_batch_custom`:
            1. Take a peak at batch_x element [1] and [2] to
                come up with class_weight for the current batch
        """
        # for adv200, context is always 1000
        df = pd.read_csv('groupings-csv/ranked_Imagenet.csv',
                 usecols=['wnid', 'idx', 'description'])
        sorted_indices = np.argsort([i for i in df['wnid']])
        indices = np.array([int(i) for i in df['idx']])[sorted_indices]

        # because adv context is always 1000 so prior IS class_weight
        weights = x[1][0, :]
        class_weight = dict(zip(indices, weights))
        # # print('\n *** class_weight at train = ', class_weight)
        # # exit()

        # --- original tf2.2 train_on_batch code --- #
        self._assert_compile_was_called()
        self._check_call_args('train_on_batch')
        with self.distribute_strategy.scope(), \
         training_utils.RespectCompiledTrainableState(self):
         iterator = data_adapter.single_batch_iterator(
                            self.distribute_strategy, 
                            x,
                            y, 
                            sample_weight,
                            class_weight)
         train_function = self.make_train_function()
         logs = train_function(iterator)

        if reset_metrics:
          self.reset_metrics()
        logs = tf_utils.to_numpy_or_python_type(logs)
        if return_dict:
          return logs
        else:
          results = [logs.get(name, None) for name in self.metrics_names]
          if len(results) == 1:
            return results[0]
          return results

    def Master_val_on_batch_custom(self, 
                                   x, 
                                   y,
                                   sample_weight=None,
                                   class_weight=None,
                                   reset_metrics=True,
                                   num_classes=None,
                                   return_dict=False):
        """
        Difference to `test_on_batch`:
            1. Take a peak at batch_x element [1] and [2] to
                come up with class_weight for the current batch

            2. This custom func is really just used at validation time.
        """
        df = pd.read_csv('groupings-csv/ranked_Imagenet.csv',
                 usecols=['wnid', 'idx', 'description'])
        sorted_indices = np.argsort([i for i in df['wnid']])
        indices = np.array([int(i) for i in df['idx']])[sorted_indices]

        weights = x[1][0, :]
        class_weight = dict(zip(indices, weights))
        # print('\n *** class_weight at val = ', class_weight)
        # exit
        # ===========================================

        self._assert_compile_was_called()
        self._check_call_args('test_on_batch')
        with self.distribute_strategy.scope():
          iterator = data_adapter.single_batch_iterator(
                            self.distribute_strategy, 
                            x,
                            y, 
                            sample_weight,
                            class_weight)  # NOTE: native tf doesn't have this option in `test_on_batch`
          test_function = self.make_test_function()
          logs = test_function(iterator)
        
        if reset_metrics:
          self.reset_metrics()
        logs = tf_utils.to_numpy_or_python_type(logs)
        if return_dict:
            return logs
        else:
          results = [logs.get(name, None) for name in self.metrics_names]
        if len(results) == 1:
          return results[0]
        return results