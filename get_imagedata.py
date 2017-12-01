"""Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.utils import np_utils
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings

from keras import backend as K

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None


def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Shifted Numpy image tensor.
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shear(x, intensity, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shear of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Sheared Numpy image tensor.
    """
    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0.):
    """Performs a random spatial zoom of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Zoomed Numpy image tensor.

    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('zoom_range should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_channel_shift(x, intensity, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x




def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.

    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x

def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        wh_tuple = (target_size[1], target_size[0])
        if img.size != wh_tuple:
            img = img.resize(wh_tuple)
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]

class ImageDataGenerator(object):
    """Generate minibatches of image data with real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 client = 1):
        if data_format is None:
            data_format = K.image_data_format()
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
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
        self.client = client

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('data_format should be "channels_last" (channel after row and '
                             'column) or "channels_first" (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('zoom_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def flow_from_directory(self, directory,sub_dir,sequence=16,
                            target_size=(256, 256), color_mode='rgb',auth_rand_train = range(1,51),
                            batch_size=32, seed=None,authentication = False,sampling = False
                            ):
        if not authentication:
            return idenDataIterator(directory,sub_dir,self,sequence=sequence,target_size=target_size,
                                    batch_size=batch_size,data_format=self.data_format,sampling=sampling)
        else:
            return authDataIterator(directory, sub_dir,self,sequence = sequence,target_size=target_size, 
                                    client=self.client,
                                    color_mode=color_mode,data_format=self.data_format,
                                    batch_size=batch_size, 
                                    seed=seed,rand_train_class=auth_rand_train)


    def img_augment(self, x,rotation = 0,h_shift=0,w_shift=0,scale=(1,1)):
        """augment a single image tensor.

        # Arguments
            x: 3D tensor, single image.

        # Returns
            A transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0

        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        # use composition of homographies
        # to generate final transform that needs to be applied
        self.rotation_range = rotation
        theta = np.pi / 180 * self.rotation_range

        self.height_shift_range = h_shift
        tx = self.height_shift_range * x.shape[img_row_axis]

        self.width_shift_range = w_shift
        ty = self.width_shift_range * x.shape[img_col_axis]

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        zx,zy = scale
        if zx != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x,
                                     self.channel_shift_range,
                                     img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)

        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        # x is a single image, so it doesn't have image number at index 0
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

        return x

    def random_transform(self, x):
        """Randomly augment a single image tensor.

        # Arguments
            x: 3D tensor, single image.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x,
                                     self.channel_shift_range,
                                     img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)

        return x


class Iterator(object):
    """Abstract base class for image data iterators.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
        sequence: len of lip sub-sequence
    """

    def __init__(self, n, sequence, seed,frame,client_len):
        self.total_images = n
        self.sequence = sequence
        self.batch_index = 0
        self.seed = seed
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.auth_index_generator = self._auth_flow_index()
        self.iden_index_generator = self._iden_flow_index()
        self.frame = frame
        self.client_num = client_len
    def reset(self):
        self.batch_index = 0

    def _auth_flow_index(self):
        # Ensure self.batch_index is 0.
        imposter_num= self.total_images - self.client_num * self.frame
        client_num = 3*self.frame
        self.reset()
        self.batch_index1 = 0
        client_array = np.arange(client_num)
        imposter_array = np.arange(imposter_num)
        imposter_flag = True
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if imposter_flag:
                current_index = (self.batch_index*self.frame) % imposter_num +\
                                (self.batch_index*self.frame) / imposter_num
                if imposter_num > current_index + self.sequence:
                    self.batch_index += 1
                else:
                    self.batch_index = 0
                    current_index = (self.batch_index * self.frame) % imposter_num +\
                                    (self.batch_index*self.frame) / imposter_num
                self.total_batches_seen += 1
                imposter_flag = False
                yield (imposter_array[current_index: current_index + self.sequence],current_index,-1)

            else:
                current_index1 = (self.batch_index1 * self.frame) % client_num + \
                                 (self.batch_index1*self.frame) / client_num
                if client_num > current_index1 + self.sequence:
                    self.batch_index1 += 1
                else:
                    self.batch_index1 = 0
                    current_index1 = (self.batch_index1 * self.frame) % client_num + \
                                     (self.batch_index1*self.frame)/client_num
                imposter_flag = True
                self.total_batches_seen += 1
                yield (client_array[current_index1: current_index1 + self.sequence],current_index1,1)
            
    def _iden_flow_index(self):
        # self.reset()
        rand_speaker_samples = np.arange(self.total_images / self.frame)
        np.random.shuffle(rand_speaker_samples[:-1])
        if len(rand_speaker_samples) % 200:
            print (rand_speaker_samples)
            raise IndexError
        clips_start = 0
        speaker_index = 0
        index_array = np.arange(self.total_images)
        while 1:

            current_index =  rand_speaker_samples[speaker_index] * self.frame + clips_start
            # current_index = (self.batch_index * self.frame) % n + (self.batch_index * self.frame) / n
            if self.total_images > current_index + self.sequence :
                if speaker_index == len(rand_speaker_samples) - 1:
                    speaker_index = 0
                    clips_start += 1
                else:
                    speaker_index += 1
            else:
                if self.seed is not None:
                    np.random.seed(self.seed + self.total_batches_seen)
                # rand_speaker_samples = np.random.shuffle(np.arange(self.total_images/self.frame))
                rand_speaker_samples = np.arange(self.total_images / self.frame)
                np.random.shuffle(rand_speaker_samples[:-1])
                # print (rand_speaker_samples)  # for debug
                clips_start = 0
                speaker_index = 0
                # current_index =  rand_speaker_samples[speaker_index] * self.frame + clips_start
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + self.sequence], current_index, 1)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

class idenDataIterator(Iterator):

    def __init__(self, directory,sub_dir, image_data_generator,sequence,
                 target_size=(100, 100), color_mode='rgb',
                 batch_size=32,seed=None,
                 data_format=None,sampling = False):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.batch_size = batch_size
        self.sequence = sequence
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.sampling = sampling
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        self.samples = 0

        classes = [str(x) for x in range(1,201)]
        self.num_class = len(classes)

        def _recursive_list(subpath):
            return sorted(os.walk(subpath), key=lambda tpl: tpl[0])
        self.sub_dir = sub_dir
        for subdir in classes:
            subpath = os.path.join(directory, subdir)   # add for
            for root, _, files in _recursive_list(subpath):
                num = int(root.split('/')[-1])
                if num in sub_dir:
                    for fname in files:
                        is_valid = False
                        for extension in white_list_formats:
                            if fname.lower().endswith('.' + extension) and not fname.lower().startswith('.'):
                                is_valid = True
                                break
                        if is_valid:
                            self.samples += 1
        print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))

        # print (classes)
        # second, build an index of the images in the different class subfolders
        self.filenames = []
        temp_filenames = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        frame = self.samples / (self.num_class*len(sub_dir))
        #print ('frame:',frame)
        i = 0
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                num = int(root.split('/')[-1])
                if num in sub_dir:
                    for fname in sorted(files):#  may have .DS_store file
                        is_valid = False
                        for extension in white_list_formats:
                            if fname.lower().endswith('.' + extension) and not fname.lower().startswith('.'):
                                is_valid = True
                                break
                        if is_valid:
                            self.classes[i] = int(subdir)-1
                            i += 1
                            # add filename relative to directory
                            absolute_path = os.path.join(root, fname)
                            temp_filenames.append(os.path.relpath(absolute_path, directory))
                    self.filenames+=sorted(temp_filenames,key=lambda x:x.split('/')[-1])
                    temp_filenames = []

        super(idenDataIterator, self).__init__(self.samples, self.sequence, seed,frame)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        sampled_sequence = self.sequence // 2 if self.sampling else self.sequence
        # print (self.sequence)
        seq_index = np.zeros((self.batch_size, sampled_sequence),dtype=int)
        seq_batch_x = np.zeros((self.batch_size,sampled_sequence)+self.image_shape,dtype=K.floatx())
        seq_batch_y = np.zeros(self.batch_size,)
        batch_x = np.zeros((sampled_sequence,) + self.image_shape, dtype=K.floatx())
        with self.lock:
            for i in range(self.batch_size):
                index_array, current_index, _ = next(self.iden_index_generator)

                fname = self.filenames[index_array[0]]
                y = int(fname.split('/')[0])
                seq_index[i] = index_array[range(0,self.sequence,2)] if self.sampling else index_array
                # print (seq_index)
                seq_batch_y[i] = y-1
        # print (seq_batch_y)
        for m in range(self.batch_size):
            wh_shift = self.image_data_generator.width_shift_range
            rand_h_shift = np.random.uniform(-wh_shift,wh_shift)
            rand_w_shift = np.random.uniform(-wh_shift,wh_shift)

            rotation = self.image_data_generator.rotation_range
            rand_rotation = np.random.uniform(-rotation,rotation)

            scale = self.image_data_generator.zoom_range
            rand_scale = np.random.uniform(scale)

            # print (seq_batch_y[m])
            # print (self.filenames[seq_index[m][0]])
            grayscale = self.color_mode == 'grayscale'

            for i, j in enumerate(seq_index[m]):
                fname = self.filenames[j]
                # print (fname)
                img = load_img(os.path.join(self.directory, fname),
                               grayscale=grayscale,
                               target_size=self.target_size)
                x = img_to_array(img, data_format=self.data_format)
                x = self.image_data_generator.img_augment(x,rotation=rand_rotation,
                                h_shift=rand_h_shift,w_shift=rand_w_shift,scale=rand_scale)
                batch_x[i] = x
            seq_batch_x[m] = batch_x
        return seq_batch_x,np_utils.to_categorical(seq_batch_y,num_classes=self.num_class)


class authDataIterator(Iterator):

    def __init__(self, directory,sub_dir,image_data_generator,client,sequence,
                 target_size=(100, 100), color_mode='rgb',rand_train_class = range(1,51),
                 batch_size=32,seed=None,
                 data_format=None):
        if data_format is None:
            data_format = K.image_data_format()
        self.client = client
        self.directory = directory
        self.batch_size = batch_size
        self.sequence = sequence
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        self.samples = 0
        self.num_class = len(rand_train_class)

        def _recursive_list(subpath):
            return sorted(os.walk(subpath), key=lambda tpl: tpl[0])
        self.client_num = len(sub_dir)
        for class_train in rand_train_class:
            subpath = os.path.join(directory, class_train)   # add for
            for root, _, files in _recursive_list(subpath):
                num = root.split('/')[-1]
                if int(num) in sub_dir:
                    for fname in files:
                        is_valid = False
                        for extension in white_list_formats:
                            if fname.lower().endswith('.' + extension):
                                is_valid = True
                                break
                        if is_valid:
                            self.samples += 1
        print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))

        # print (classes)
        # second, build an index of the images in the different class subfolders
        self.filenames = []
        temp_filenames = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        frame = self.samples / (self.num_class*self.client_num)
        #print ('frame:',frame)
        i = 0
        for class_train in rand_train_class:
            subpath = os.path.join(directory, class_train)
            for root, _, files in _recursive_list(subpath):
                num = root.split('/')[-1]
                if int(num) in sub_dir:
                    for fname in sorted(files):#  may have .DS_store file
                        is_valid = False
                        for extension in white_list_formats:
                            if fname.lower().endswith('.' + extension):
                                is_valid = True
                                break
                        if is_valid:
                            self.classes[i] = int(class_train)-1
                            i += 1
                            # add filename relative to directory
                            absolute_path = os.path.join(root, fname)
                            temp_filenames.append(os.path.relpath(absolute_path, directory))
                    self.filenames+=sorted(temp_filenames,key=lambda x:int(x[-8:-4]))
                    temp_filenames = []
        print ([self.filenames[x] for x in range(1,10000,100)])
        super(authDataIterator, self).__init__(self.samples, self.sequence, seed,frame,self.client_num)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        seq_index = np.zeros((self.batch_size, self.sequence),dtype=int)
        seq_batch_x = np.zeros((self.batch_size,self.sequence)+self.image_shape,dtype=K.floatx())
        seq_batch_y = np.zeros(self.batch_size,)
        batch_x = np.zeros((self.sequence,) + self.image_shape, dtype=K.floatx())
        with self.lock:
            for i in range(self.batch_size):
                index_array, current_index, y = next(self.auth_index_generator)
                if y==-1:
                    index_array = index_array + self.client_num * self.frame
                fname = self.filenames[index_array[0]]
                y= int(fname.split('/')[0])
                seq_index[i] = index_array
                seq_batch_y[i] = 1 if y==self.client else 0
                #print (seq_batch_y)

        for m in range(self.batch_size):
            wh_shift = self.image_data_generator.width_shift_range
            rand_h_shift = np.random.uniform(-wh_shift, wh_shift)
            rand_w_shift = np.random.uniform(-wh_shift, wh_shift)

            rotation = self.image_data_generator.rotation_range
            rand_rotation = np.random.uniform(-rotation, rotation)

            scale = self.image_data_generator.zoom_range
            rand_scale = np.random.uniform(scale)

            # print (seq_batch_y[m])
            # print (self.filenames[seq_index[m][0]])
            grayscale = self.color_mode == 'grayscale'

            for i, j in enumerate(seq_index[m]):
                fname = self.filenames[j]
                # print (fname)
                img = load_img(os.path.join(self.directory, fname),
                               grayscale=grayscale,
                               target_size=self.target_size)
                x = img_to_array(img, data_format=self.data_format)
                x = self.image_data_generator.img_augment(x, rotation=rand_rotation,
                                        h_shift=rand_h_shift, w_shift=rand_w_shift, scale=rand_scale)
                batch_x[i] = x
            seq_batch_x[m] = batch_x
        return seq_batch_x,np_utils.to_categorical(seq_batch_y,num_classes=2)



if __name__=='__main__':

    index_generator = Iterator(600,2,10,3).auth_index_generator
    while 1:
        print (next(index_generator))