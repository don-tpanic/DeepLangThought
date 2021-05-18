import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import copy
import warnings
import threading
import multiprocessing
import numpy as np
import tensorflow as tf
from TRAIN.utils.data_utils import load_classes


class DirectoryIterator(object):
    white_list_formats = {'tfrecords'}

    def __init__(self,
                 directory,
                 classes=None,
                 seed=42,
                 follow_links=False,
                 subset='validation',
                 validation_split=0.1,
                 sup=None,
                 ):

        self.classes = classes
        self.validation_split = validation_split
        self.directory = directory

        if subset is not None:
            validation_split = self.validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError(
                    'Invalid subset name: %s;'
                    'expected "training" or "validation"' % (subset,))
        else:
            split = None

        self.split = split
        self.subset = subset

        if validation_split and not 0 < validation_split < 1:
            raise ValueError(
                '`validation_split` must be strictly between 0 and 1. '
                ' Received: %s' % validation_split)
        self.validation_split = validation_split

        # First, count the number of samples and classes.
        self.samples = 0
        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    # classes here is a list of all wnids
                    classes.append(subdir)
        self.num_classes = len(classes)
        # mapping from wnid to class indices as integers
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()
        # Second, build an index of the images
        # in the different class subfolders.
        results = []
        self.filenames = []
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(
                pool.apply_async(_list_valid_filenames_in_directory,
                                 (dirpath, self.white_list_formats, self.split,
                                  self.class_indices, follow_links)))
        classes_list = []
        for res in results:
            classes, filenames = res.get()
            classes_list.append(classes)
            self.filenames += filenames
        self.samples = len(self.filenames)
        self.classes = np.zeros((self.samples,), dtype='int32')

        # now the self.classes is a list of integer class indices
        for classes in classes_list:
            self.classes[i:i + len(classes)] = classes
            i += len(classes)

        print('Total images [%d] from [%d] classes' %
              (self.samples, self.num_classes))

        pool.close()
        pool.join()
        # Until this point, everthing is default.
        # Now if we aim for coarsegrain training,
        # We need to set up a different mapping `class_indices_sup`
        #######################################################################################
        # HACK: modify class_indices to have super groups
        # and then we will have to keep track of both the native class labels 
        # and the hacked superGroup labels.
        print(f'[Check] sup is [{sup}]')
        if sup is None:
            # It can really be anything, but has to be defined
            # otherwise, it will fail check later when setting up batch_y
            self.class_indices_sup = None
        else:
            self.class_indices_sup = copy.deepcopy(self.class_indices)
            sup_wnids, sup_indices, sup_descriptions = load_classes(num_classes=999, df=sup)
            print(f'[Check] sup set size = {len(sup_wnids)}')
            for wnid in self.class_indices:
                # NOTE(ken): here which label we set to depends on
                # if we fine tune the VGG output or not.
                # if finetune, we could set the label to be the last index
                # if not, we could set the label to be the first dog's index.
                if wnid in sup_wnids:
                    # now all dogs are swapped with the same label
                    self.class_indices_sup[wnid] = sup_indices[0]
            #print(self.class_indices)
            #print(self.class_indices_sup)

            # HACK: we continue the hack here by creating on parallel all examples indices 
            # that is according to the class_indices_sup, basically we repeat the above process
            # but using `class_indices_sup`
            
            # NOTE: classes overloaded, at this point, classes are already overidden.
            # Thus, here if we use all 1k classes, we can just redo what has been done above.
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    # classes here is a list of all wnids
                    classes.append(subdir)

            pool = multiprocessing.pool.ThreadPool()
            # Second, build an index of the images
            # in the different class subfolders.
            results = []
            self.filenames = []
            i = 0
            for dirpath in (os.path.join(directory, subdir) for subdir in classes):
                results.append(
                    pool.apply_async(_list_valid_filenames_in_directory,
                                    (dirpath, self.white_list_formats, self.split,
                                    self.class_indices_sup, follow_links)))
            classes_list = []
            for res in results:
                classes, filenames = res.get()
                classes_list.append(classes)
                self.filenames += filenames
            self.samples = len(self.filenames)
            self.classes_sup = np.zeros((self.samples,), dtype='int32')
            # now the self.classes is a list of integer class indices
            for classes in classes_list:
                self.classes_sup[i:i + len(classes)] = classes
                i += len(classes)

            print('CHECK: before subsample, total images [%d] from [%d] classes' %
                (self.samples, self.num_classes))

            pool.close()
            pool.join()
        # The hack until this point.
        #######################################################################################
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]


    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        return self.classes

    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None


def _count_valid_files_in_directory(directory,
                                    white_list_formats,
                                    split,
                                    follow_links):
    num_files = len(list(
        _iter_valid_files(directory, white_list_formats, follow_links)))
    if split:
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
    else:
        start, stop = 0, num_files
    return stop - start


def _list_valid_filenames_in_directory(directory, white_list_formats, split,
                                       class_indices, follow_links):
    dirname = os.path.basename(directory)
    if split:
        num_files = len(list(
            _iter_valid_files(directory, white_list_formats, follow_links)))
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = list(
            _iter_valid_files(
                directory, white_list_formats, follow_links))[start: stop]
    else:
        valid_files = _iter_valid_files(
            directory, white_list_formats, follow_links)

    classes = []
    filenames = []
    for root, fname in valid_files:
        classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)

    return classes, filenames


def _iter_valid_files(directory, white_list_formats, follow_links):
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links),
                      key=lambda x: x[0])

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            for extension in white_list_formats:
                if fname.lower().endswith('.tiff'):
                    warnings.warn('Using \'.tiff\' files with multiple bands '
                                  'will cause distortion. '
                                  'Please verify your output.')
                if fname.lower().endswith('.' + extension):
                    yield root, fname