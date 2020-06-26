"""
What is this script:
--------------------
    (current version: 1/8/2019) the create_good_generator is built on keras.utils.Sequence
    which is now thread-safe.
"""
import numpy as np
import pandas as pd

from tensorflow.keras.applications.vgg16 import preprocess_input

from .labels_corrector import wnids_to_network_indices, indices_rematch
#
###
from . import native_sequence_generator as Gen
from . import tf_custom_gen_using_sequence as SafeGen
from . import continuous_master_gen_sequence as MasterGen
from . import continuous_master_testTime_gen as MasterGenTEST
from . import batchMomentum_gen


#
### This is for fine tune VGG16 using Sequence
def simple_generator(directory,
                     classes,
                     batch_size,
                     seed,
                     shuffle,
                     subset,
                     validation_split,
                     class_mode,
                     target_size,
                     preprocessing_function,
                     horizontal_flip,
                     AlexNetAug=False,
                     ):
    """
    Nothing special about this generator,
    i.e. no branches, no pseudo input, no momentum
    """
    if classes == None:
        pass
    else:
        sorted_classes = sorted(classes)

    # the initial generator
    # using Sequence
    bad_generator = Gen.SafeDirectoryIterator(
                        directory=directory,
                        classes=classes,
                        batch_size=batch_size,
                        seed=seed,
                        shuffle=shuffle,
                        subset=subset,
                        validation_split=validation_split,
                        class_mode=class_mode,
                        target_size=target_size,
                        preprocessing_function=preprocessing_function,
                        horizontal_flip=horizontal_flip,
                        )

    steps = bad_generator.compute_step_size()

    # label correction
    if classes == None:
        # when use all 1000 categories, there is no need to rematch
        # keras-auto labelled indices to the real network indices
        # because keras labels all categories in the order of wnids which is
        # the same as network indices
        # so the bad_generator is already index correct!
        index_correct_generator = bad_generator
    else:
        # Sanity check: network_indices are also sorted in ascending order
        network_indices = wnids_to_network_indices(sorted_classes)

        # rematch indices and get the index_correct_generator
        index_correct_generator = indices_rematch(bad_generator, network_indices)

    good_generator = index_correct_generator
    return good_generator, steps

#
### Can be used up until training continuous-master model (exclusive)
### This is the ICML generator + continuous attention generator.
def create_good_generator(
                          directory,
                          classes,
                          batch_size,
                          seed,
                          shuffle,
                          subset,
                          validation_split,
                          class_mode,
                          target_size,
                          preprocessing_function,
                          horizontal_flip,
                          AlexNetAug=False,
                          focus_classes=None,
                          subsample_rate=1
                          ):
    """
    usage (current version - only available on ken's branch):
    ------
        Now the generator is created as an instance of the Sequence class as
        the recommended keras.utils.Sequence to avoid unsafe multi-threading.

        The difference compared to the previous version is that we do all data loading
        and data augmentation at one go avoiding creating wrapper class `ImageDataGenerator`
        which isn't an instance of Sequence.

        `SafeDirectoryIterator` will take all params at once. You can find its implementation
        in `custom_gen_using_sequence.py`. Essentially we removed one extra class (wrapper)
        such as instance of `SafeDirectoryIterator` is an instance of Sequence.

    Notes (current version):
    ------
        AlexNetAug is suspended.

    return:
    -------
        - a generator which can be used in fitting
        - steps that is required when evaluating

    Example:
    --------
        Say you want to train model on categories ['dog', 'cat', 'ball'] which have
        wordnet ids ['n142', 'n99', 'n200'] and their real indices on VGG's output layer
        are [234, 101, 400]. The function works as follows:

            1. You pass in classes=['n142', 'n99', 'n200']
            2. classes will be sorted as ['n99', 'n142', 'n200']
            3. keras auto-label them as [0, 1, 2]
            4. `index_correct_generator` will relabel three categories as [101, 234, 400]
            5. use extra Alexnet augmentation if specified.
    """

    '''
    # why sort classes?
    -------------------
        sort wordnet ids alphabatically (may not be necessary)
        if sorted, keras will label the smallest wordnet id as class 0, so on.
        and in the future when we need to replace class 0 with the actual network
        index, class 0 will be replaced with the smallest network index as it should
        be in sync with wordnet ids which are sorted in the first place.
    '''
    if classes == None:
        pass
    else:
        sorted_classes = sorted(classes)

    # the initial generator
    # using Sequence
    bad_generator = SafeGen.SafeDirectoryIterator(directory=directory,
                                                    classes=classes,
                                                    batch_size=batch_size,
                                                    seed=seed,
                                                    shuffle=shuffle,
                                                    subset=subset,
                                                    validation_split=validation_split,
                                                    class_mode=class_mode,
                                                    target_size=target_size,
                                                    preprocessing_function=preprocessing_function,
                                                    horizontal_flip=horizontal_flip,
                                                    focus_classes=focus_classes,
                                                    subsample_rate=subsample_rate
                                                    )
    # print('next bad_generator')
    # bad_generator.__next__()
    # help(bad_generator)   # NOTE (28/03/2020): this is the custom gen, which uses the commened next method!

    # number of steps go through the dataset is a required parameter later
    # WARNING: comment out for compute_step_size() in sumsample cases
    # steps = np.ceil(len(bad_generator.classes) / batch_size)
    steps = bad_generator.compute_step_size()

    # label correction
    if classes == None:
        # when use all 1000 categories, there is no need to rematch
        # keras-auto labelled indices to the real network indices
        # because keras labels all categories in the order of wnids which is
        # the same as network indices
        # so the bad_generator is already index correct!
        index_correct_generator = bad_generator
    else:
        # Sanity check: network_indices are also sorted in ascending order
        network_indices = wnids_to_network_indices(sorted_classes)

        # rematch indices and get the index_correct_generator
        index_correct_generator = indices_rematch(bad_generator, network_indices)


    good_generator = index_correct_generator
    return good_generator, steps

#
### Only can be used for training-validation of continuous-master model
def generator_for_continuous_master(directory,
                                    classes,
                                    batch_size,
                                    seed,
                                    shuffle,
                                    subset,
                                    validation_split,
                                    class_mode,
                                    target_size,
                                    preprocessing_function,
                                    horizontal_flip,
                                    # ---------------------
                                    # for master model only
                                    focus_indices,
                                    intensities,
                                    # ---------------------
                                    AlexNetAug=False,
                                    focus_classes=None,
                                    subsample_rate=1,
                                    ):
    """
        Customised only for training+validation the continuous_master model;
        and CANNOT be used for testing due to different requirements (see below)

        Update (05/04/2020): here validation set is randomly changing at the end of 
        each training epoch. To have the same validation set, use `generator_for_continuous_master_VAL`
    """
    if classes == None:
        pass
    else:
        sorted_classes = sorted(classes)

    # the initial generator
    # using Sequence
    bad_generator = MasterGen.SafeDirectoryIterator(directory=directory,
                                                    classes=classes,
                                                    batch_size=batch_size,
                                                    seed=seed,
                                                    shuffle=shuffle,
                                                    subset=subset,
                                                    validation_split=validation_split,
                                                    class_mode=class_mode,
                                                    target_size=target_size,
                                                    preprocessing_function=preprocessing_function,
                                                    horizontal_flip=horizontal_flip,
                                                    focus_classes=focus_classes,
                                                    subsample_rate=subsample_rate,
                                                    # ----------------------------
                                                    # master model only
                                                    focus_indices=focus_indices,
                                                    intensities=intensities)

    # number of steps go through the dataset is a required parameter later
    # WARNING: comment out for compute_step_size() in sumsample cases
    # steps = np.ceil(len(bad_generator.classes) / batch_size)
    steps = bad_generator.compute_step_size()

    # label correction
    if classes == None:
        # when use all 1000 categories, there is no need to rematch
        # keras-auto labelled indices to the real network indices
        # because keras labels all categories in the order of wnids which is
        # the same as network indices
        # so the bad_generator is already index correct!
        index_correct_generator = bad_generator
    else:
        # Sanity check: network_indices are also sorted in ascending order
        network_indices = wnids_to_network_indices(sorted_classes)

        # rematch indices and get the index_correct_generator
        index_correct_generator = indices_rematch(bad_generator, network_indices)

    good_generator = index_correct_generator
    return good_generator, steps

#
### Only can be used for testing of continuous-master model
def generator_for_continuous_master_TEST(directory,
                                        classes,
                                        batch_size,
                                        seed,
                                        shuffle,
                                        subset,
                                        validation_split,
                                        class_mode,
                                        target_size,
                                        preprocessing_function,
                                        horizontal_flip,
                                        # ---------------------
                                        # for master model only (TEST)
                                        usr_focus_index,      # REVIEW: a user suppplied class index
                                        usr_intensity,        # REVIEW: a user supplied intensity
                                        num_of_classes,       # REVIEW: a user supplied total number of focus classes.
                                        # ---------------------
                                        AlexNetAug=False,
                                        focus_classes=None,
                                        subsample_rate=1,
                                        ):
    """
    Only use for testing for a given unique combo of context pair supplied by
    user.
    """
    if classes == None:
        pass
    else:
        sorted_classes = sorted(classes)
    bad_generator = MasterGenTEST.SafeDirectoryIterator(directory=directory,
                                                        classes=classes,
                                                        batch_size=batch_size,
                                                        seed=seed,
                                                        shuffle=shuffle,
                                                        subset=subset,
                                                        validation_split=validation_split,
                                                        class_mode=class_mode,
                                                        target_size=target_size,
                                                        preprocessing_function=preprocessing_function,
                                                        horizontal_flip=horizontal_flip,
                                                        focus_classes=focus_classes,
                                                        subsample_rate=subsample_rate,
                                                        # ----------------------------
                                                        # master model only (TEST)
                                                        focus_index=usr_focus_index,
                                                        intensity=usr_intensity,
                                                        num_of_classes=num_of_classes,
                                                        )
    steps = bad_generator.compute_step_size()

    # label correction
    if classes == None:
        index_correct_generator = bad_generator
    else:
        network_indices = wnids_to_network_indices(sorted_classes)
        index_correct_generator = indices_rematch(bad_generator, network_indices)

    good_generator = index_correct_generator
    return good_generator, steps

#
###
# temp: batchMomentum experiment:
def generator_for_batchMomentum(directory,
                                    classes,
                                    batch_size,
                                    seed,
                                    shuffle,
                                    subset,
                                    validation_split,
                                    class_mode,
                                    target_size,
                                    preprocessing_function,
                                    horizontal_flip,
                                    # ---------------------
                                    # for master model only
                                    focus_indices,
                                    intensities,
                                    batchMomentum,
                                    # ---------------------
                                    AlexNetAug=False,
                                    focus_classes=None,
                                    subsample_rate=1,
                                    ):
    """
    Customised only for training+validation the continuous_master model;
    and CANNOT be used for testing due to different requirements (see below)
    Update (05/04/2020): here validation set is randomly changing at the end of 
    each training epoch. To have the same validation set, use `generator_for_continuous_master_VAL`
    """
    if classes == None:
        pass
    else:
        sorted_classes = sorted(classes)

    # the initial generator
    # using Sequence
    bad_generator = batchMomentum_gen.SafeDirectoryIterator(directory=directory,
                                                    classes=classes,
                                                    batch_size=batch_size,
                                                    seed=seed,
                                                    shuffle=shuffle,
                                                    subset=subset,
                                                    validation_split=validation_split,
                                                    class_mode=class_mode,
                                                    target_size=target_size,
                                                    preprocessing_function=preprocessing_function,
                                                    horizontal_flip=horizontal_flip,
                                                    focus_classes=focus_classes,
                                                    subsample_rate=subsample_rate,
                                                    # ----------------------------
                                                    # master model only
                                                    focus_indices=focus_indices,
                                                    intensities=intensities,
                                                    batchMomentum=batchMomentum
                                                    )

    # number of steps go through the dataset is a required parameter later
    # WARNING: comment out for compute_step_size() in sumsample cases
    # steps = np.ceil(len(bad_generator.classes) / batch_size)
    steps = bad_generator.compute_step_size()

    # label correction
    if classes == None:
        # when use all 1000 categories, there is no need to rematch
        # keras-auto labelled indices to the real network indices
        # because keras labels all categories in the order of wnids which is
        # the same as network indices
        # so the bad_generator is already index correct!
        index_correct_generator = bad_generator
    else:
        # Sanity check: network_indices are also sorted in ascending order
        network_indices = wnids_to_network_indices(sorted_classes)

        # rematch indices and get the index_correct_generator
        index_correct_generator = indices_rematch(bad_generator, network_indices)

    good_generator = index_correct_generator
    return good_generator, steps