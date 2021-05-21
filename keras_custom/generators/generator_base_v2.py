import os
import copy
import warnings
import threading
import multiprocessing
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array, apply_affine_transform, apply_channel_shift, apply_brightness_shift

from TRAIN.utils.data_utils import load_classes
from . import simclr_preprocessing

"""
Simclr generator is used to preprocess images based on Simclr routine,
where we use preprocess defined in `simclr_preprocessing` replacing 
all pre-fined augmentations here. 

One limitation is that previously, native generator does everything in 
numpy space whereas simclr preprocessing does everything in native tensor
space, so we have to do two conversions:
    1. convert the loaded image into tensor to be processed by simclr preprocessing 
    2. convert the preprocessed version back to ndarray so the model can work with it.
"""

class Iterator(Sequence):
    """
    An iterator inherits the keras.utils.Sequence as recommended for
    multiprocessing safety
    """
    white_list_formats = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff')

    def __init__(self, n, batch_size, shuffle, seed, wordvec_mtx, sup):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()
        # custom stuff
        self.wordvec_mtx = wordvec_mtx
        self.sup = sup

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            # np.random.seed(self.seed)
            self.index_array = np.random.permutation(self.n)

    def compute_step_size(self):
        if self.index_array is None:
            self._set_index_array()   # QUESTION:effect?
        len_of_index_array = len(self.index_array)
        step_size = np.ceil(len_of_index_array / self.batch_size)
        return step_size

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        # NOTE(ken): wordvec_mtx: must do self. otherwise undefined due to __getitem__
        return self._get_batches_of_transformed_samples(index_array, 
                                                        self.wordvec_mtx, 
                                                        self.sup)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        print(f'[Check] _flow_index')
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            if self.n == 0:
                current_index = 0
            else:
                current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        print(f'[Check] next')
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array, self.wordvec_mtx)

    def _get_batches_of_transformed_samples(self, index_array, wordvec_mtx, sup):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: Array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


class DirectoryIterator(Iterator):
    """
    Custom DirectoryIterator work with CustomIterator
    (flow_from_directory uses this class, where flow_from_directory is
    a function under class ImageDataGenerator)
    """
    allowed_class_modes = {'categorical', 'binary', 'sparse', 'input', None}

    def __init__(self,
                 directory,
                 target_size=(224, 224),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype=None,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 preprocessing_function=None,
                 validation_split=0.0,
                 interpolation_order=1,
                 # custom attributes -->
                 wordvec_mtx=None,
                 simclr_range=False,
                 simclr_augment=False,
                 sup=None,
                 ):

        if color_mode not in {'rgb', 'rgba', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "rgba", or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        self.target_size = tuple(target_size)

        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        self.classes = classes
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
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
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.preprocessing_function = preprocessing_function
        self.dtype = dtype
        self.wordvec_mtx = wordvec_mtx
        self.simclr_range = simclr_range
        self.simclr_augment = simclr_augment

        if dtype is None:
            self.dtype = K.floatx()
        self.interpolation_order = interpolation_order
        self.data_format = data_format
        
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2
        if validation_split and not 0 < validation_split < 1:
            raise ValueError(
                '`validation_split` must be strictly between 0 and 1. '
                ' Received: %s' % validation_split)
        self.validation_split = validation_split

        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))
        self.class_mode = class_mode
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

        super(DirectoryIterator, self).__init__(self.samples,
                                                batch_size,
                                                shuffle,
                                                seed,
                                                wordvec_mtx,
                                                sup)

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

    def _get_batches_of_transformed_samples(self, index_array, wordvec_mtx, sup):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: Array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        for i, j in enumerate(index_array):

            # PIL.Image
            img = load_img(filepaths[j],
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            # ndarray, [1, 255]
            x = img_to_array(img, data_format=self.data_format)

            ### ###
            # WARNING: For now stop using this, simply use x/255. for simclr.
            # NOTE(ken)
            # convert to tensor so can be preprocessed by simclr 
            # _preprocess
            # and then convert back to array
            if self.simclr_range:
                # x = x / 255.
                x = tf.convert_to_tensor(x, dtype=tf.uint8)
                x = simclr_preprocessing._preprocess(x, is_training=self.simclr_augment)
            # NOTE(ken) convert back to numpy or not 
            # both work n no influence on training time.
            # x = x.numpy()
            ### ###

            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            if self:
                params = self.get_random_transform(x.shape)
                x = self.apply_transform(x, params)
                x = self.standardize(x)
            batch_x[i] = x

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.
        elif self.class_mode == 'multi_output':
            batch_y = [output[index_array] for output in self.labels]
        elif self.class_mode == 'raw':
            batch_y = self.labels[index_array]
        else:
            return batch_x
        
        # -------------------------------------------------------------------------
        # NOTE(ken): Nick's trick to get semantic targets 
        # if mtx is not provided, we return only the labels.
        if wordvec_mtx is not None:
            # if finegrain
            semantic_truth = np.dot(batch_y, wordvec_mtx)
            if sup is None:       
                batch_y = [semantic_truth, batch_y]
            # if coarsegrain
            else:
                # get the coarsegrain labels.
                batch_y_sup = np.zeros((len(batch_x), len(self.class_indices_sup)),
                                dtype=self.dtype)
                for i, n_observation in enumerate(index_array):
                    batch_y_sup[i, self.classes_sup[n_observation]] = 1.
                batch_y = [semantic_truth, batch_y_sup]
        # -------------------------------------------------------------------------
        if self.sample_weight is None:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.sample_weight[index_array]

    def standardize(self, x):
        """Applies the normalization configuration in-place to a batch of inputs.
        `x` is changed in-place since the function is mainly used internally
        to standarize images and feed them to your network. If a copy of `x`
        would be created instead it would have a significant performance cost.
        If you want to apply this method without changing the input in-place
        you can call the method creating a copy before:
        standarize(np.copy(x))
        # Arguments
            x: Batch of inputs to be normalized.
        # Returns
            The inputs, normalized.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        return x

    def get_random_transform(self, img_shape, seed=None):
        """Generates random parameters for a transformation.
        # Arguments
            seed: Random seed.
            img_shape: Tuple of integers.
                Shape of the image that is transformed.
        # Returns
            A dictionary containing randomly chosen parameters describing the
            transformation.
        """
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1

        if seed is not None:
            np.random.seed(seed)

        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self.vertical_flip

        transform_parameters = {'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical,
                                }
        return transform_parameters

    def apply_transform(self, x, transform_parameters):
        """Applies a transformation to an image according to given parameters.
        # Arguments
            x: 3D tensor, single image.
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.
                Currently, the following parameters
                from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip.
                - `'flip_vertical'`: Boolean. Vertical flip.
                - `'channel_shift_intencity'`: Float. Channel shift intensity.
                - `'brightness'`: Float. Brightness shift intensity.
        # Returns
            A transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        x = apply_affine_transform(x, transform_parameters.get('theta', 0),
                                   transform_parameters.get('tx', 0),
                                   transform_parameters.get('ty', 0),
                                   transform_parameters.get('shear', 0),
                                   transform_parameters.get('zx', 1),
                                   transform_parameters.get('zy', 1),
                                   row_axis=img_row_axis,
                                   col_axis=img_col_axis,
                                   channel_axis=img_channel_axis,
                                   fill_mode=self.fill_mode,
                                   cval=self.cval,
                                   order=self.interpolation_order)

        if transform_parameters.get('flip_horizontal', False):
            x = flip_axis(x, img_col_axis)

        if transform_parameters.get('flip_vertical', False):
            x = flip_axis(x, img_row_axis)

        return x

    def random_transform(self, x, seed=None):
        """Applies a random transformation to an image.
        # Arguments
            x: 3D tensor, single image.
            seed: Random seed.
        # Returns
            A randomly transformed version of the input (same shape).
        """
        params = self.get_random_transform(x.shape, seed)
        return self.apply_transform(x, params)


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


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
