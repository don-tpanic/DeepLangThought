from .labels_corrector import wnids_to_network_indices, indices_rematch
from . import superGroupGenerator as supGen
from . import generator_base, generator_base_v2

"""
Customer generators wrapper for Language model only.
"""

def sup_gen(directory,
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
                wordvec_mtx,
                sup
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
    bad_generator = supGen.SafeDirectoryIterator(
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
                        wordvec_mtx=wordvec_mtx,
                        sup=sup
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


def data_generator_v2(directory,
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
                    wordvec_mtx,
                    simclr_range=False,
                    simclr_augment=False,
                    sup=None):
    """
    Purpose:
    --------
        v2. supports both finegrain n coarsegrain training.
    """
    if classes == None:
        pass
    else:
        sorted_classes = sorted(classes)

    # the initial generator
    bad_generator = generator_base_v2.DirectoryIterator(
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
                        wordvec_mtx=wordvec_mtx,
                        simclr_range=simclr_range,
                        simclr_augment=simclr_augment,
                        sup=sup)
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
