import os
import warnings
import threading
import multiprocessing
from functools import partial

import numpy as np
import matplotlib
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array, apply_affine_transform, apply_channel_shift, apply_brightness_shift

"""
Purpose:
--------
    14/05/2020: `custom_generator` will produce `generator_for_continuous_master`
                using the Sequence class here and return the final generator.
"""

class SafeIterator(Sequence):
    """
        An iterator inherits the keras.utils.Sequence as recommended for
        multiprocessing safety
    """
    white_list_formats = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff')
    def __init__(self, n, batch_size, shuffle, seed, subsample_weights, all_labels, focus_indices, intensities):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

        # WARNING: which gets overriden in the inherited class using super
        self._subsample_weights = subsample_weights
        self._classes = all_labels

        # ----------------------------------
        self._focus_indices = focus_indices
        self._intensities = intensities
        self.focus_index_this_epoch = None
        self.intensity_this_epoch = None

    # subsample in progress
    def _set_index_array(self):
        """
            1. called every epoch
            2. all classes are first loaded and than subsampled.
        """
        ####################################################################################################
        subsample_weights = self._subsample_weights  # dict
        all_labels = self._classes   # list of ints
        unique_classes = np.unique(all_labels)

        # NOTE: if not, n gets overriden and the entire set isn't the full size
        index_array = np.arange(len(all_labels))  # n is total num of imgs found
        print('CHECK: before subsample has [%s] images' % len(index_array))

        subsample_dict = {}
        for class_i in unique_classes:
            locs = np.where(all_labels==class_i)
            subsample_dict[class_i] = index_array[locs]

        # second, subsample based on weights
        new_index_array = []
        for class_i in subsample_dict:
            weight = subsample_weights[class_i]
            if weight == 1:
                new_index_array = np.hstack((new_index_array, subsample_dict[class_i]))
            else:
                num_in_class = len(subsample_dict[class_i])
                num_in_subsample = int(np.ceil(weight * num_in_class))

                # NOTE: can't seed because we want different samples each at each epoch
                subsample = np.random.choice(subsample_dict[class_i], size=num_in_subsample, replace=False)
                new_index_array = np.hstack((new_index_array, subsample))
        ####################################################################################################

        new_index_array = np.array([int(i) for i in new_index_array])
        print('CHECK: subsampled [%s] images ' % len(new_index_array))
        if self.shuffle:
            new_index_array = np.random.permutation(new_index_array)
        self.index_array = new_index_array
        self.n = len(new_index_array)

    def compute_step_size(self):
        if self.index_array is None:
            self._set_index_array()   # # QUESTION:effect?
        len_of_index_array = len(self.index_array)
        step_size = np.ceil(len_of_index_array / self.batch_size)
        return step_size

    def get_epoch_labels(self):

        # WARNING: may not be accurate due to separetly calling `_set_index_array`
        # WARNING: will result in changing the index array due to random choice.

        # original labels for all before subsample
        all_labels = self._classes
        if self.index_array is None:
            self._set_index_array()
        epoch_labels = all_labels[self.index_array]
        return epoch_labels
    ############################################################################

    def __getitem__(self, idx):
        """
            Called every batch.
        """
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
        
        # self.total_batches_seen_per_epoch += 1
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        # consistent with len(new_index_array)
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            if self.n == 0:
                # Avoiding modulo by zero error
                current_index = 0
            else:
                current_index = (self.batch_index * self.batch_size) % self.n
                # print('current_index = %s' % current_index)
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                # print('batch_index reset to 0.')
                self.batch_index = 0
            self.total_batches_seen += 1
            # print('yielding batch index...')
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
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: Array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


class SafeDirectoryIterator(SafeIterator):
    """
        Custom DirectoryIterator work with CustomIterator
        (flow_from_directory uses this class, where flow_from_directory is
        a function under class ImageDataGenerator)
    """
    allowed_class_modes = {'categorical', 'binary', 'sparse', 'input', None}

    def __init__(self,
                 directory,
                 # image_data_generator,
                 target_size=(256, 256),
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
                 dtype=None,                                                    # WARNING:  original is 32
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 validation_split=0.0,
                 interpolation_order=1,
                 # ---------------------------
                 focus_classes=None,
                 subsample_rate=1,
                 ### for continuous master ####
                 focus_indices=None,
                 intensities=None,
                 # ---------------------------
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

        self.color_mode = color_mode
        self.data_format = data_format
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

        ########################################################################
        # stuff from previous ImageDataGenerator
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.dtype = dtype

        # --------------------------------------------------------
        if dtype is None:
            self.dtype = K.floatx()
        # --------------------------------------------------------

        self.interpolation_order = interpolation_order

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError(
                '`data_format` should be `"channels_last"` '
                '(channel after row and column) or '
                '`"channels_first"` (channel before row and column). '
                'Received: %s' % data_format)
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

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received: %s' % (zoom_range,))
        if zca_whitening:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, which overrides '
                              'setting of `featurewise_center`.')
            if featurewise_std_normalization:
                self.featurewise_std_normalization = False
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening` '
                              'which overrides setting of'
                              '`featurewise_std_normalization`.')
        if featurewise_std_normalization:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, '
                              'which overrides setting of '
                              '`featurewise_center`.')
        if samplewise_std_normalization:
            if not samplewise_center:
                self.samplewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`samplewise_std_normalization`, '
                              'which overrides setting of '
                              '`samplewise_center`.')
        if brightness_range is not None:
            if (not isinstance(brightness_range, (tuple, list)) or
                    len(brightness_range) != 2):
                raise ValueError(
                    '`brightness_range should be tuple or list of two floats. '
                    'Received: %s' % (brightness_range,))
        self.brightness_range = brightness_range

        #### End of code from previous ImageDataGenerator ####
        ########################################################################


        ################
        self.focus_indices = focus_indices
        self.intensities = intensities
        # ----------------------------------
        self.focus_classes = focus_classes
        self.subsample_rate = subsample_rate
        # can be none or
        # a list of wnids: focus_classes = ['n01234']
        ################

        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = dtype
        #######
        if dtype is None:
            self.dtype = K.floatx()
        #######


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

        ########################################################################
        # subsample_weights
        subsample_weights = {}
        if self.focus_classes is None:
            for wnid in self.class_indices:
                subsample_weights[self.class_indices[wnid]] = 1.

        else:
            for wnid in self.class_indices:
                if wnid in self.focus_classes:
                    # the focus classes don't subsample or downweight
                    subsample_weights[self.class_indices[wnid]] = 1.
                else:
                    # out of context will be subsampled.
                    # when subsample_rate = 0, is same as not using out context at all
                    subsample_weights[self.class_indices[wnid]] = self.subsample_rate
        ########################################################################

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

        print('CHECK: before subsample, total images [%d] from [%d] classes' %
              (self.samples, self.num_classes))

        pool.close()
        pool.join()
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]

        ####### self.samples = self.n in upper class #######
        super(SafeDirectoryIterator, self).__init__(self.samples,
                                                    batch_size,
                                                    shuffle,
                                                    seed,
                                                    subsample_weights,
                                                    self.classes,
                                                    # -----------------
                                                    self.focus_indices,
                                                    self.intensities
                                                    )
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

    def _get_batches_of_transformed_samples(self, index_array):
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
            img = load_img(filepaths[j],
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
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


        ###### build second and third input to attention factory ###############
        # a random context pair is sampled for a batch
        self.focus_index_this_epoch = np.random.choice(self._focus_indices, size=1, replace=True)
        self.intensity_this_epoch = np.random.choice(self._intensities, size=1, replace=True)
        
        # NOTE: v2 - prior is the Brad prior; intensity in (?, 1)
        # NOTE/TEMP: for adv training, the context size is always 1000 if to match ICML
        # the only difference is focus_indices are among 200 not 1000 classes. 
        non_trg_int = (1 - self.intensity_this_epoch) / 999
        prior = np.ones((batch_x.shape[0], 1000)) * non_trg_int
        prior[:, self.focus_index_this_epoch] = self.intensity_this_epoch

        intensity_this_epoch = np.ones((batch_x.shape[0], 1)) * self.intensity_this_epoch
        batch_x = [batch_x, prior, intensity_this_epoch]
        ########################################################################
        if self.sample_weight is None:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.sample_weight[index_array]

    ############################################################################
    #### Below are methods in previous ImageDataGenerator ####
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
        if self.rescale:
            x *= self.rescale
        if self.samplewise_center:
            x -= np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, keepdims=True) + 1e-6)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-6)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, '
                              'but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
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

        if self.rotation_range:
            theta = np.random.uniform(
                -self.rotation_range,
                self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            try:  # 1-D array-like or int
                tx = np.random.choice(self.height_shift_range)
                tx *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                tx = np.random.uniform(-self.height_shift_range,
                                       self.height_shift_range)
            if np.max(self.height_shift_range) < 1:
                tx *= img_shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            try:  # 1-D array-like or int
                ty = np.random.choice(self.width_shift_range)
                ty *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                ty = np.random.uniform(-self.width_shift_range,
                                       self.width_shift_range)
            if np.max(self.width_shift_range) < 1:
                ty *= img_shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(
                -self.shear_range,
                self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(
                self.zoom_range[0],
                self.zoom_range[1],
                2)

        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self.vertical_flip

        channel_shift_intensity = None
        if self.channel_shift_range != 0:
            channel_shift_intensity = np.random.uniform(-self.channel_shift_range,
                                                        self.channel_shift_range)

        brightness = None
        if self.brightness_range is not None:
            brightness = np.random.uniform(self.brightness_range[0],
                                           self.brightness_range[1])



        ########################################################################
        # pca_augment parameters
        pca_std_deviation = 0.1
        pca_scale = 1.0
        ########################################################################

        transform_parameters = {'theta': theta,
                                'tx': tx,
                                'ty': ty,
                                'shear': shear,
                                'zx': zx,
                                'zy': zy,
                                'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical,
                                'channel_shift_intensity': channel_shift_intensity,
                                'brightness': brightness,
                                'pca_std_deviation': pca_std_deviation, ### pca aug
                                'pca_scale': pca_scale   ### pca aug
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

        if transform_parameters.get('channel_shift_intensity') is not None:
            x = apply_channel_shift(x,
                                    transform_parameters['channel_shift_intensity'],
                                    img_channel_axis)

        if transform_parameters.get('flip_horizontal', False):
            x = flip_axis(x, img_col_axis)

        if transform_parameters.get('flip_vertical', False):
            x = flip_axis(x, img_row_axis)

        if transform_parameters.get('brightness') is not None:
            x = apply_brightness_shift(x, transform_parameters['brightness'])


        ########################################################################
        # pca transform function
        # x = pca_augment(x,
        #                 transform_parameters['pca_std_deviation'],
        #                 transform_parameters['pca_scale'])
        ########################################################################


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

    #### Above are methods implemented in the previous ImageDataGenerator ####
    ############################################################################


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def pca_augment(inputs, std_deviation=0.1, scale=1.0, clipping=False):
    """
    usage:
    ------
        AlexNet PCA augmentation
    references:
    -----------
        https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        https://blog.shikoan.com/pca-color-augmentation/
    """
    ranks = inputs.ndim
    assert ranks >= 2

    chs = inputs.shape[-1]

    # swapaxis, reshape for calculating covariance matrix
    # rank 2 = (batch, dims)
    # rank 3 = (batch, step, dims)
    if ranks <= 3:
        x = inputs.copy()
    # rank 4 = (batch, height, width, ch) -> (batch, dims, ch)
    elif ranks == 4:
        dims = inputs.shape[1] * inputs.shape[2]
        x = inputs.reshape(-1, dims, chs)
    # rank 5 = (batch, D, H, W, ch) -> (batch, D, dims, ch)
    elif ranks == 5:
        dims = inputs.shape[2] * inputs.shape[3]
        depth = inputs.shape[1]
        x = inputs.reshape(-1, depth, dims, chs)

    # scaling-factor
    calculate_axis, reduce_axis = ranks-1, ranks-2
    if ranks == 3:
        calculate_axis, reduce_axis = 1, 2
    elif ranks >= 4:
        calculate_axis, reduce_axis = ranks-3, ranks-2
    C = 1.0
    if ranks >= 3:
        C = x.shape[reduce_axis]

    ###########################################################################
    ### normalize x by using mean and std
    # variance within each chl
    var = np.var(x, axis=calculate_axis, keepdims=True)
    # 1./std along each chl
    scaling_factors = np.sqrt(C / np.sum(var, axis=reduce_axis, keepdims=True))
    # scaling
    x = x * scaling_factors
    # subtract mean for cov matrix
    mean = np.mean(x, axis=calculate_axis, keepdims=True)
    x -= mean
    ###########################################################################
    # covariance matrix
    cov_n = max(x.shape[calculate_axis] - 1, 1)
    # cov (since x was normalized --> x.T * x gives the var-cov matrix)
    cov = np.matmul(np.swapaxes(x, -1, -2), x) / cov_n

    # eigen value(S), eigen vector(U)
    U, S, V = np.linalg.svd(cov)

    # random values
    # if rank2 : get differnt random variable by sample
    if ranks == 2:
        rand = np.random.randn(*inputs.shape) * std_deviation
        delta = np.matmul(rand*np.expand_dims(S, axis=0), U)
    else:
        # rand -> size=len(S), random int between low and high eigenvalues, multiply std
        rand = np.random.randn(*S.shape) * std_deviation
        # [p1, p2, p3][a1r1, a2r2, a3r3].T
        delta_original = np.squeeze(np.matmul(U, np.expand_dims(rand*S, axis=-1)), axis=-1)

    # adjust delta shape
    if ranks == 3:
        delta = np.expand_dims(delta_original, axis=ranks-2)
    elif ranks >= 4:
        delta = np.expand_dims(delta_original, axis=ranks-3)
        delta = np.broadcast_to(delta, x.shape)
        delta = delta.reshape(-1, *inputs.shape[1:])

    # delta scaling
    delta = delta * scale

    result = inputs + delta
    if clipping:
        """
        vgg16 does not clip:
        https://arxiv.org/pdf/1409.1556.pdf
        """
        result = np.clip(result, 0.0, scale)

    return result


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
