# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Transforms used in the PBA Augmentation Policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import inspect
import random

import numpy as np
from PIL import ImageOps, ImageEnhance, ImageFilter, Image
import model_aug_config

PARAMETER_MAX = 10  # What is the max 'level' a transform could be predicted


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
    return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
    return int(level * maxval / PARAMETER_MAX)


class TransformFunction(object):
    """Wraps the Transform function for pretty printing options."""

    def __init__(self, func, name):
        self.f = func
        self.name = name

    def __repr__(self):
        return '<' + self.name + '>'

    def __call__(self, pil_img):
        return self.f(pil_img)


def _rotate_impl(pil_img, level):
    """Rotates `pil_img` from -config.rotate_max_degree to config.rotate_max_degree degrees depending on `level`."""
    degrees = int_parameter(level, model_aug_config.rotate_max_degree)
    if random.random() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees)


def _posterize_impl(pil_img, level):
    """Applies PIL Posterize to `pil_img`."""
    level = int_parameter(level, model_aug_config.posterize_max)
    return ImageOps.posterize(pil_img.convert('RGB'),
                              model_aug_config.posterize_max - level)

def _shear_x_impl(pil_img, level, image_size):
    """Applies PIL ShearX to `pil_img`.

  The ShearX operation shears the image along the horizontal axis with `level`
  magnitude.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had ShearX applied to it.
  """
    level = float_parameter(level, model_aug_config.shear_x_max)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((image_size, image_size), Image.AFFINE, (1, level, 0, 0, 1, 0))


def _shear_y_impl(pil_img, level, image_size):
    """Applies PIL ShearY to `pil_img`.

  The ShearY operation shears the image along the vertical axis with `level`
  magnitude.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had ShearX applied to it.
  """
    level = float_parameter(level, model_aug_config.shear_y_max)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((image_size, image_size), Image.AFFINE, (1, 0, 0, level, 1, 0))


def _translate_x_impl(pil_img, level, image_size):
    """Applies PIL TranslateX to `pil_img`.

  Translate the image in the horizontal direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had TranslateX applied to it.
  """
    level = int_parameter(level, model_aug_config.translate_x_max)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((image_size, image_size), Image.AFFINE, (1, 0, level, 0, 1, 0))


def _translate_y_impl(pil_img, level, image_size):
    """Applies PIL TranslateY to `pil_img`.

  Translate the image in the vertical direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had TranslateY applied to it.
  """
    level = int_parameter(level, model_aug_config.translate_x_max)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((image_size, image_size), Image.AFFINE, (1, 0, 0, 0, 1, level))


def _crop_impl(pil_img, level, image_size, interpolation=Image.BILINEAR):
    """Applies a crop to `pil_img` with the size depending on the `level`."""
    cropped = pil_img.crop((level, level, image_size - level,
                            image_size - level))
    resized = cropped.resize((image_size, image_size), interpolation)
    return resized


def _solarize_impl(pil_img, level):
    """Applies PIL Solarize to `pil_img`.

  Translate the image in the vertical direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had Solarize applied to it.
  """
    level = int_parameter(level, 256)
    return ImageOps.solarize(pil_img.convert('RGB'),
                             256 - level)


def _cutout_pil_impl(pil_img, level, image_size, mean):
    """Apply cutout to pil_img at the specified level."""
    cutout_mean = [int(255*x) for x in mean]
    size = int_parameter(level, model_aug_config.cutout_max_size)
    if size <= 0:
        return pil_img
    img_height, img_width, num_channels = (image_size, image_size, 3)
    _, upper_coord, lower_coord = (create_cutout_mask(img_height, img_width,
                                                      num_channels, size))
    pixels = pil_img.load()  # create the pixel map
    for i in range(upper_coord[0], lower_coord[0]):  # for every col:
        for j in range(upper_coord[1], lower_coord[1]):  # For every row
            pixels[i, j] = (cutout_mean[0], cutout_mean[1], cutout_mean[2], 0)  # set the colour accordingly
    return pil_img


def _enhancer_impl(enhancer):
    """Sets level to be between 0.1 and config.enhance_max for ImageEnhance transforms of PIL."""

    def impl(pil_img, level):
        v = float_parameter(level, model_aug_config.enhance_max) + .1  # going to 0 just destroys it
        return enhancer(pil_img).enhance(v)

    return impl


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def pil_transformer(self, probability, level, image_size, mean):
        """Builds augmentation function which returns resulting image and whether augmentation was applied."""

        def return_function(im):
            res = False
            if random.random() < probability:
                args = inspect.getargspec(self.xform).args
                if 'image_size' in args:
                    if 'mean' in args:
                        im = self.xform(im, level, image_size, mean)
                    else:
                        im = self.xform(im, level, image_size)
                else:
                    im = self.xform(im, level)
                res = True
            return im, res

        name = self.name + '({:.1f},{})'.format(probability, level)
        return TransformFunction(return_function, name)

    def str(self):
        return self.name


def create_cutout_mask(img_height, img_width, num_channels, size):
    """Creates a zero mask used for cutout of shape `img_height` x `img_width`.

  Args:
    img_height: Height of image cutout mask will be applied to.
    img_width: Width of image cutout mask will be applied to.
    num_channels: Number of channels in the image.
    size: Size of the zeros mask.

  Returns:
    A mask of shape `img_height` x `img_width` with all ones except for a
    square of zeros of shape `size` x `size`. This mask is meant to be
    elementwise multiplied with the original image. Additionally returns
    the `upper_coord` and `lower_coord` which specify where the cutout mask
    will be applied.
  """
    assert img_height == img_width

    # Sample center where cutout mask will be applied
    height_loc = np.random.randint(low=0, high=img_height)
    width_loc = np.random.randint(low=0, high=img_width)

    # Determine upper right and lower left corners of patch
    upper_coord = (max(0, height_loc - size // 2), max(0,
                                                       width_loc - size // 2))
    lower_coord = (min(img_height, height_loc + size // 2),
                   min(img_width, width_loc + size // 2))
    mask_height = lower_coord[0] - upper_coord[0]
    mask_width = lower_coord[1] - upper_coord[1]
    assert mask_height > 0
    assert mask_width > 0

    mask = np.ones((img_height, img_width, num_channels))
    zeros = np.zeros((mask_height, mask_width, num_channels))
    mask[upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1], :] = (
        zeros)
    return mask, upper_coord, lower_coord


################## Transform Functions ##################
identity = TransformT('identity', lambda pil_img, level: pil_img)
flip_lr = TransformT(
    'FlipLR', lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))
flip_ud = TransformT(
    'FlipUD', lambda pil_img, level: pil_img.transpose(Image.FLIP_TOP_BOTTOM))
# pylint:disable=g-long-lambda
auto_contrast = TransformT(
    'AutoContrast',
    lambda pil_img, level: ImageOps.autocontrast(pil_img.convert('RGB'))
)
equalize = TransformT(
    'Equalize',
    lambda pil_img, level: ImageOps.equalize(pil_img.convert('RGB'))
)
invert = TransformT(
    'Invert',
    lambda pil_img, level: ImageOps.invert(pil_img.convert('RGB'))
)
# pylint:enable=g-long-lambda
blur = TransformT('Blur',
                  lambda pil_img, level: pil_img.filter(ImageFilter.BLUR))
smooth = TransformT('Smooth',
                    lambda pil_img, level: pil_img.filter(ImageFilter.SMOOTH))
rotate = TransformT('Rotate', _rotate_impl)
posterize = TransformT('Posterize', _posterize_impl)
shear_x = TransformT('ShearX', _shear_x_impl)
shear_y = TransformT('ShearY', _shear_y_impl)
translate_x = TransformT('TranslateX', _translate_x_impl)
translate_y = TransformT('TranslateY', _translate_y_impl)
crop_bilinear = TransformT('CropBilinear', _crop_impl)
solarize = TransformT('Solarize', _solarize_impl)
cutout = TransformT('Cutout', _cutout_pil_impl)
color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))
contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
brightness = TransformT('Brightness', _enhancer_impl(ImageEnhance.Brightness))
sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))

ALL_TRANSFORMS = [
    flip_lr, flip_ud, auto_contrast, equalize, invert, rotate, posterize,
    crop_bilinear, solarize, color, contrast, brightness, sharpness, shear_x,
    shear_y, translate_x, translate_y, cutout, blur, smooth
]

NAME_TO_TRANSFORM = {t.name: t for t in ALL_TRANSFORMS}


def apply_policy(policy, pil_img, mean, image_size, verbose=False):
    """Apply the `policy` to the numpy `img`.
  Args:
    policy: A list of tuples with the form (name, probability, level) where
      `name` is the name of the augmentation operation to apply, `probability`
      is the probability of applying the operation and `level` is what strength
      the operation to apply.
    pil_img: Numpy image that will have `policy` applied to it.
    mean: channel mean of dataset
    image_size: Width and height of image.
    verbose: Whether to print applied augmentations.
  Returns:
    The result of applying `policy` to `img`.
  """
    count = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.5, 0.0])
    if count != 0:
        policy = copy.copy(policy)
        random.shuffle(policy)
        for xform in policy:
            assert len(xform) == 3
            name, probability, level = xform
            assert 0. <= probability <= 1.
            assert 0 <= level <= PARAMETER_MAX
            xform_fn = NAME_TO_TRANSFORM[name].pil_transformer(probability, level, image_size, mean)
            pil_img, res = xform_fn(pil_img)
            if verbose and res:
                print("Op: {}, Magnitude: {}, Prob: {}".format(name, level, probability))
            count -= res
            assert count >= 0
            if count == 0:
                break
        return pil_img
    else:
        return pil_img
