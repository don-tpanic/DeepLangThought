import os
import warnings
import numpy as np
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.callbacks import ModelCheckpoint


# custom ModelCheckpoint: save attention weights on the end of every epoch
# so that we can create gif over attention weights change over epochs
class ModelCheckpoint_and_save_attention(Callback):

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, category=None):  ###category is new, as in category
        super(ModelCheckpoint_and_save_attention, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        # custom added
        self.category = category

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            print('pseudo saving best model weights...')
                            pass
                            # self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))

                #---------------------------------------------------
                # save attention weights on every epoch's end
                # att_ws = self.model.get_layer('att_layer_1').get_weights()[0]
                # fig, ax = plt.subplots()
                # ax.set_ylim([0, 4])
                # plt.scatter(range(len(att_ws)), att_ws, alpha=0.2)
                #
                # directory = 'dynamic_attention/%s/' % self.category
                # if not os.path.exists(directory):
                #     os.makedirs(directory)
                # plt.savefig(directory + 'attention_batch%s_epoch%s.png' % (self.category, epoch))
                # plt.close()
                #---------------------------------------------------

            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


# custom stopping criteria: relative improvement in percentage
class RelativeEarlyStopping(Callback):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 min_perc_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(RelativeEarlyStopping, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.min_perc_delta = min_perc_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf


    def on_epoch_end(self, epoch, logs=None):

        current = self.get_monitor_value(logs)
        if current is None:
            return

        print('\n\n')
        relative_change = (abs(self.best - current) / self.best) * 100
        print('----------------------------')
        print('relative change = {%.3f} percent' % relative_change)
        print('----------------------------')
        print('best = %s' % self.best)
        print('current = %s' % current)
        print('----------------------------')
        print('\n\n')


        if self.monitor_op(current, self.best * (1 - self.min_perc_delta)):  # only this line matters

            print('### still improves over criterion ###')

            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:

            print('### wait += 1 ###')

            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value


class Checkpoint_AttnFactory(ModelCheckpoint):

    def __init__(self,
               filepath,
               monitor='val_loss',
               verbose=0,
               save_best_only=True,
               save_weights_only=True,
               mode='min',
               save_freq='epoch',  # TODO: custom freq.
               factory_depth=1,          # later supplied by user.
               run=None,           # later supplied by user.
               **kwargs):

        super(Checkpoint_AttnFactory, self).__init__(filepath)

        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.epochs_since_last_save = 0
        self.factory_depth = factory_depth
        self.run = run

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            # only save attn factory weights
                            for i in range(1, self.factory_depth+1):
                                hid_weights = self.model.get_layer('hidden_layer_%s' % i).get_weights()
                                kernel = hid_weights[0]
                                bias = hid_weights[1]

                                np.save('attention_weights/continuous_master/partial_weights/adv/hid{i}_kernel_run{run}.npy'.format(i=i, run=self.run), kernel)
                                np.save('attention_weights/continuous_master/partial_weights/adv/hid{i}_bias_run{run}.npy'.format(i=i, run=self.run), bias)

                            attn_weights = self.model.get_layer('produce_attention_weights').get_weights()
                            attn_factory_kernel = attn_weights[0]
                            np.save('attention_weights/continuous_master/partial_weights/adv/attn_factory_kernel_run{run}.npy'.format(run=self.run), attn_factory_kernel)
                            print('((( factory weights saved. )))')
                        else:
                            # This will cause problem when having custom
                            # layer etc. And not recommended to save
                            # the entire model.
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True) 


class LangTotalLoss(Callback):
    """
    Monitoring semantic+descreite loss
    of the lang model.
    """

    # https://stackoverflow.com/questions/59336883/only-show-total-loss-during-training-of-a-multi-output-model-in-keras
    # https://keras.io/guides/writing_your_own_callbacks/
    
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):

        super(LangTotalLoss, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        """
        monitor total loss and terminate given % improve
        """
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        if self.restore_best_weights:
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights:
                if self.verbose > 0:
                    print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)

        # we want to return the total loss at validation
        # and we want to print total loss at train epoch end
        # val_semantic_loss = logs['val_semantic_loss']
        # val_discrete_loss = logs['val_discrete_loss']

        if monitor_value is None:
            logging.warning('Early stopping conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
        return monitor_value
