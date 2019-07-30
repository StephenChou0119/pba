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
"""Data utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import copy
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from pba.cutout import Cutout
import torchvision

from pba.csv_dataset import CsvDataset
try:
    import cPickle as pickle
except:
    import pickle
import tensorflow as tf
from torch.utils.data import DataLoader
from pba.utils import parse_log_schedule
import random

import numpy as np
from PIL import ImageOps, ImageEnhance, ImageFilter, Image
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


class DataSet(object):
    """Dataset object that produces augmented training and eval data.
    Load raw data from hack dataset.

        Assumes data is in NHWC format.
        Args:
            hparams: tf.hparams object.
    """

    def __init__(self, hparams):
        self.__cutout_size = hparams.cutout_size
        self.__num_workers = hparams.num_workers
        self.__padding_size = hparams.padding_size
        self.image_size = hparams.size_of_image
        self.num_classes = hparams.num_of_classes
        self.hparams = hparams
        self.HP_TRANSFORM_NAMES = hparams.HP_TRANSFORM_NAMES
        self.NUM_HP_TRANSFORM = hparams.NUM_HP_TRANSFORM
        self.use_pba = hparams.use_pba
        self.mean, self.std = hparams.mean, hparams.std
        self.__pba_tansform_object = self.PbaTransform(hparams.rotate_max_degree, hparams.posterize_max, \
                                                       hparams.enhance_max, hparams.shear_x_max, hparams.shear_y_max, \
                                                       hparams.translate_x_max, hparams.translate_y_max, hparams.cutout_max_size)
        if self.use_pba:
            self.parse_policy(hparams)
        self.__pba_transform = transforms.Lambda(lambda img: self._apply_pba(img) if self.use_pba else img)
        self.__normalize = transforms.Normalize(self.mean, self.std)
        self.__test_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                self.__normalize,
            ]
        )
        self.reset_dataloader()
        self.__curr_epoch = 0

    class PbaTransform:
        """

        Args:
            rotate_max_degree: max value of rotate, rotate range from [-max,max]
            posterize_max: max value of posterize, posterize range from [0, max]
            enhance_max: max value of enhancements(color contrast brightness sharpness), enchance range from [0.1, max+0.1]
            shear_x_max: max value of shear_x, shear_x range from [-max, max]
            shear_y_max: max value of shear_y, shear_y range from [-max, max]
            translate_x_max: max value of translate_x, translate_x range from [-max, max]
            translate_y_max: max value of translate_y, translate_y range from [-max, max]
            cutout_max_size: max value of cutout, [0,max]
        """
        def __init__(self, rotate_max_degree=30, posterize_max=4,\
                     enhance_max=1.8,shear_x_max=0.3,shear_y_max=0.3,\
                     translate_x_max=10,translate_y_max=10,cutout_max_size=20):
            self.__rotate_max_degree = rotate_max_degree
            self.__posterize_max = posterize_max
            self.__enhance_max = enhance_max
            self.__shear_x_max = shear_x_max
            self.__shear_y_max = shear_y_max
            self.__translate_x_max = translate_x_max
            self.__translate_y_max = translate_y_max
            self.__cutout_max_size = cutout_max_size
            self.identity = TransformT('identity', lambda pil_img, level: pil_img)
            self.flip_lr = TransformT(
                'FlipLR', lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))
            self.flip_ud = TransformT(
                'FlipUD', lambda pil_img, level: pil_img.transpose(Image.FLIP_TOP_BOTTOM))
            # pylint:disable=g-long-lambda
            self.auto_contrast = TransformT(
                'AutoContrast',
                lambda pil_img, level: ImageOps.autocontrast(pil_img.convert('RGB'))
            )
            self.equalize = TransformT(
                'Equalize',
                lambda pil_img, level: ImageOps.equalize(pil_img.convert('RGB'))
            )
            self.invert = TransformT(
                'Invert',
                lambda pil_img, level: ImageOps.invert(pil_img.convert('RGB'))
            )
            # pylint:enable=g-long-lambda
            self.blur = TransformT('Blur',
                              lambda pil_img, level: pil_img.filter(ImageFilter.BLUR))
            self.smooth = TransformT('Smooth',
                                lambda pil_img, level: pil_img.filter(ImageFilter.SMOOTH))
            self.rotate = TransformT('Rotate', self._rotate_impl)
            self.posterize = TransformT('Posterize', self._posterize_impl)
            self.shear_x = TransformT('ShearX', self._shear_x_impl)
            self.shear_y = TransformT('ShearY', self._shear_y_impl)
            self.translate_x = TransformT('TranslateX', self._translate_x_impl)
            self.translate_y = TransformT('TranslateY', self._translate_y_impl)
            self.crop_bilinear = TransformT('CropBilinear', self._crop_impl)
            self.solarize = TransformT('Solarize', self._solarize_impl)
            self.cutout = TransformT('Cutout', self._cutout_pil_impl)
            self.color = TransformT('Color', self._enhancer_impl(ImageEnhance.Color))
            self.contrast = TransformT('Contrast', self._enhancer_impl(ImageEnhance.Contrast))
            self.brightness = TransformT('Brightness',self. _enhancer_impl(ImageEnhance.Brightness))
            self.sharpness = TransformT('Sharpness', self._enhancer_impl(ImageEnhance.Sharpness))

            self.ALL_TRANSFORMS = [
                self.flip_lr, self.flip_ud, self.auto_contrast, self.equalize, self.invert, self.rotate, self.posterize,
                self.crop_bilinear, self.solarize, self.color, self.contrast, self.brightness, self.sharpness, self.shear_x,
                self.shear_y, self.translate_x, self.translate_y, self.cutout, self.blur, self.smooth
            ]

            self.NAME_TO_TRANSFORM = {t.name: t for t in self.ALL_TRANSFORMS}

        def _rotate_impl(self, pil_img, level):
            """Rotates `pil_img` from -config.rotate_max_degree to config.rotate_max_degree degrees depending on `level`."""
            degrees = int_parameter(level, self.__rotate_max_degree)
            if random.random() > 0.5:
                degrees = -degrees
            return pil_img.rotate(degrees)

        def _posterize_impl(self, pil_img, level):
            """Applies PIL Posterize to `pil_img`."""
            level = int_parameter(level, self.__posterize_max)
            return ImageOps.posterize(pil_img.convert('RGB'),
                                      self.__posterize_max - level)

        def _shear_x_impl(self, pil_img, level, image_size):
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
            level = float_parameter(level, self.__shear_x_max)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform((image_size, image_size), Image.AFFINE, (1, level, 0, 0, 1, 0))

        def _shear_y_impl(self, pil_img, level, image_size):
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
            level = float_parameter(level, self.__shear_y_max)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform((image_size, image_size), Image.AFFINE, (1, 0, 0, level, 1, 0))

        def _translate_x_impl(self, pil_img, level, image_size):
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
            level = int_parameter(level, self.__translate_x_max)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform((image_size, image_size), Image.AFFINE, (1, 0, level, 0, 1, 0))

        def _translate_y_impl(self, pil_img, level, image_size):
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
            level = int_parameter(level, self.__translate_x_max)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform((image_size, image_size), Image.AFFINE, (1, 0, 0, 0, 1, level))

        def _crop_impl(self, pil_img, level, image_size, interpolation=Image.BILINEAR):
            """Applies a crop to `pil_img` with the size depending on the `level`."""
            cropped = pil_img.crop((level, level, image_size - level,
                                    image_size - level))
            resized = cropped.resize((image_size, image_size), interpolation)
            return resized

        def _solarize_impl(self, pil_img, level):
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

        def _cutout_pil_impl(self, pil_img, level, image_size, mean):
            """Apply cutout to pil_img at the specified level."""
            cutout_mean = [int(255 * x) for x in mean]
            size = int_parameter(level, self.__cutout_max_size)
            if size <= 0:
                return pil_img
            img_height, img_width, num_channels = (image_size, image_size, 3)
            _, upper_coord, lower_coord = (self.create_cutout_mask(img_height, img_width,
                                                              num_channels, size))
            pixels = pil_img.load()  # create the pixel map
            for i in range(upper_coord[0], lower_coord[0]):  # for every col:
                for j in range(upper_coord[1], lower_coord[1]):  # For every row
                    pixels[i, j] = (cutout_mean[0], cutout_mean[1], cutout_mean[2], 0)  # set the colour accordingly
            return pil_img

        def _enhancer_impl(self, enhancer):
            """Sets level to be between 0.1 and config.enhance_max for ImageEnhance transforms of PIL."""

            def impl(pil_img, level):
                v = float_parameter(level, self.__enhance_max) + .1  # going to 0 just destroys it
                return enhancer(pil_img).enhance(v)

            return impl

        def create_cutout_mask(self, img_height, img_width, num_channels, size):
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

        def apply_policy(self, policy, pil_img, mean, image_size, verbose=False):
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
                    xform_fn = self.NAME_TO_TRANSFORM[name].pil_transformer(probability, level, image_size, mean)
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
    @property
    def curr_epoch(self):
        return self.__curr_epoch

    @curr_epoch.setter
    def curr_epoch(self, current_epoch):
        if not isinstance(current_epoch, int):
            raise ValueError('epoch must be an integer!')
        if current_epoch < 0:
            raise ValueError('epoch must > 0!')
        self.__curr_epoch = current_epoch

    def reset_dataloader(self, ):
        """
        reset transform after reset policy
        Returns:

        """

        if self.hparams.dataset_type == 'custom':
            tf.logging.info('using custom config!')
            self.__train_data_root = self.hparams.train_data_root
            self.__train_csv_path = self.hparams.train_csv_path
            self.__val_data_root = self.hparams.val_data_root
            self.__val_csv_path = self.hparams.val_csv_path
            self.__test_data_root = self.hparams.test_data_root
            self.__test_csv_path = self.hparams.test_csv_path
            crop = transforms.Lambda(lambda img: TF.crop(img, 251 - 250, 273 - 250, 500, 500))
            self.__transform = transforms.Compose(
                [
                    crop,
                    transforms.Resize((self.image_size, self.image_size)),
                    self.__pba_transform,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=(self.image_size, self.image_size), padding=self.__padding_size),
                    transforms.ToTensor(),
                    self.__normalize,
                    Cutout(n_holes=1, length=self.__cutout_size)
                ]
            )
            self.train_loader = DataLoader(
                CsvDataset(self.__train_data_root, self.__train_csv_path, transform=self.__transform),
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=self.__num_workers,
                drop_last=True)
            self.val_loader = DataLoader(
                CsvDataset(self.__val_data_root, self.__val_csv_path, transform=self.__test_transform),
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.__num_workers,
                drop_last=True)
            self.test_loader = DataLoader(
                CsvDataset(self.__test_data_root, self.__test_csv_path, transform=self.__test_transform),
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.__num_workers,
                drop_last=True)
        elif self.hparams.dataset_type == 'cifar10':
            self.data_root = self.hparams.data_root
            self.__transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    self.__pba_transform,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=(self.image_size, self.image_size), padding=self.__padding_size),
                    transforms.ToTensor(),
                    self.__normalize,
                    Cutout(n_holes=1, length=self.__cutout_size)
                ]
            )
            self.train_loader = DataLoader(
                torchvision.datasets.CIFAR10(self.data_root, train=True, transform=self.__transform, download=True),
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=self.__num_workers,
                drop_last=True)
            self.val_loader = DataLoader(
                torchvision.datasets.CIFAR10(self.data_root, train=False, transform=self.__test_transform, download=True),
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.__num_workers,
                drop_last=True
            )
            self.test_loader = DataLoader(
                torchvision.datasets.CIFAR10(self.data_root, train=False, transform=self.__test_transform, download=True),
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.__num_workers,
                drop_last=True
            )
        elif self.hparams.dataset_type == 'svhn':
            self.data_root = self.hparams.data_root
            self.__transform = transforms.Compose(
                [
                    self.__pba_transform,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=(self.image_size, self.image_size), padding=self.__padding_size),
                    transforms.ToTensor(),
                    self.__normalize,
                    Cutout(n_holes=1, length=self.__cutout_size)
                ]
            )
            self.train_loader = DataLoader(
                torchvision.datasets.SVHN(self.data_root, split='train', transform=self.__transform, download=True),
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=self.__num_workers,
                drop_last=True)
            self.val_loader = DataLoader(
                torchvision.datasets.SVHN(self.data_root, split='test', transform=self.__test_transform, download=True),
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.__num_workers,
                drop_last=True
            )
            self.test_loader = DataLoader(
                torchvision.datasets.SVHN(self.data_root, split='test', transform=self.__test_transform, download=True),
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.__num_workers,
                drop_last=True
            )
        self.__train_iter = iter(self.train_loader)

        self.num_train = len(self.train_loader)
        self.num_val = len(self.val_loader)
        self.num_test = len(self.test_loader)
        tf.logging.info('reset data loader!')

    def _apply_pba(self, data):
        """

        Args:
            data: pillow image object

        Returns:
            pillow image object
        """
        # apply PBA policy)
        if isinstance(self.policy[0], list):
            # single policy
            final_img = self.__pba_tansform_object.apply_policy(
                self.policy[self.curr_epoch],
                data,
                self.mean,
                image_size=self.image_size)
        elif isinstance(self.policy, list):
            # policy schedule
            final_img = self.__pba_tansform_object.apply_policy(
                self.policy,
                data,
                self.mean,
                image_size=self.image_size)
        else:
            raise ValueError('Unknown policy.')
        return final_img

    def parse_policy(self, hparams):
        """Parses policy schedule from input, which can be a list, list of lists, text file, or pickled list.
        If list is not nested, then uses the same policy for all epochs.
        Args:
        hparams: tf.hparams object.
        """
        # Parse policy
        if isinstance(hparams.hp_policy,
                      str) and hparams.hp_policy.endswith('.txt'):
            if hparams.num_epochs % hparams.hp_policy_epochs != 0:
                tf.logging.warning(
                    "Schedule length (%s) doesn't divide evenly into epochs (%s), interpolating.",
                    hparams.num_epochs, hparams.hp_policy_epochs)
            tf.logging.info(
                'schedule policy trained on {} epochs, parsing from: {}, multiplier: {}'
                .format(
                    hparams.hp_policy_epochs, hparams.hp_policy,
                    float(hparams.num_epochs) / hparams.hp_policy_epochs))
            raw_policy = parse_log_schedule(
                hparams.hp_policy,
                epochs=hparams.hp_policy_epochs,
                multiplier=float(hparams.num_epochs) /
                hparams.hp_policy_epochs)
        elif isinstance(hparams.hp_policy, list):
            # support list of hp_policy for search stage
            raw_policy = hparams.hp_policy
        else:
            raise ValueError('hp_policy must be txt or None during training!')
        if isinstance(raw_policy[0], list):
            self.policy = []
            split = len(raw_policy[0]) // 2
            for pol in raw_policy:
                cur_pol = self._parse_policy(pol[:split])
                cur_pol.extend(
                    self._parse_policy(pol[split:]))
                self.policy.append(cur_pol)
            tf.logging.info('using HP policy schedule, last: {}'.format(
                self.policy[-1]))
        elif isinstance(raw_policy, list):
            split = len(raw_policy) // 2
            self.policy = self._parse_policy(raw_policy[:split])
            self.policy.extend(
                self._parse_policy(raw_policy[split:]))
            tf.logging.info('using HP Policy, policy: {}'.format(
                self.policy))

    def _parse_policy(self, policy_emb):
        """parse list policy"""
        policy = []
        num_xform = self.NUM_HP_TRANSFORM
        xform_names = self.HP_TRANSFORM_NAMES
        assert len(policy_emb
                   ) == 2 * num_xform, 'policy was: {}, supposed to be: {}'.format(
            len(policy_emb), 2 * num_xform)
        for i, xform in enumerate(xform_names):
            policy.append((xform, policy_emb[2 * i] / 10., policy_emb[2 * i + 1]))
        return policy

    def reset_policy(self, new_hparams):
        """
        reset policy after an epoch start and before transform has been reset.
        Args:
            new_hparams:

        Returns:

        """
        self.hparams = new_hparams
        self.parse_policy(new_hparams)
        tf.logging.info('reset aug policy')
        return

    def next_batch(self):
        """Return the next minibatch of augmented data."""
        try:
            batched_data = next(self.__train_iter)
        except StopIteration:
            pass
        # 1. torch.Tensor to np.array
        images, labels = batched_data
        images = images.numpy()
        labels = labels.numpy()
        # 2. label to one hot
        labels = np.eye(self.num_classes)[labels]
        batched_data = (np.array(images, np.float32), labels)
        return batched_data


class TransformFunction(object):
    """Wraps the Transform function for pretty printing options."""

    def __init__(self, func, name):
        self.f = func
        self.name = name

    def __repr__(self):
        return '<' + self.name + '>'

    def __call__(self, pil_img):
        return self.f(pil_img)


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