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

import copy
from pba.csv_dataset import CsvDataset
try:
    import cPickle as pickle
except:
    import pickle
import os
import random
import numpy as np
import tensorflow as tf
import torchvision
from torch.utils.data import DataLoader

from autoaugment.data_utils import unpickle
import pba.policies as found_policies
from pba.utils import parse_log_schedule
import pba.augmentation_transforms_hp as augmentation_transforms_pba
import pba.augmentation_transforms as augmentation_transforms_autoaug


# pylint:disable=logging-format-interpolation


def parse_policy(policy_emb, augmentation_transforms):
    policy = []
    num_xform = augmentation_transforms.NUM_HP_TRANSFORM
    xform_names = augmentation_transforms.HP_TRANSFORM_NAMES
    assert len(policy_emb
               ) == 2 * num_xform, 'policy was: {}, supposed to be: {}'.format(
                   len(policy_emb), 2 * num_xform)
    for i, xform in enumerate(xform_names):
        policy.append((xform, policy_emb[2 * i] / 10., policy_emb[2 * i + 1]))
    return policy


def shuffle_data(data, labels):
    """Shuffle data using numpy."""
    np.random.seed(0)
    perm = np.arange(len(data))
    np.random.shuffle(perm)
    data = data[perm]
    labels = labels[perm]
    return data, labels


class DataSet(object):
    """Dataset object that produces augmented training and eval data.
    """

    def __init__(self, hparams):
        self.hparams = hparams
        self.epochs = 0

        self.parse_policy(hparams)
        self.load_data(hparams)

        # Apply normalization
        self.train_images = self.train_images.transpose(0, 2, 3, 1) / 255.0
        self.val_images = self.val_images.transpose(0, 2, 3, 1) / 255.0
        self.test_images = self.test_images.transpose(0, 2, 3, 1) / 255.0
        mean = augmentation_transforms_autoaug.dataset_mean
        std = augmentation_transforms_autoaug.dataset_mean
        tf.logging.info('mean:{}    std: {}'.format(mean, std))

        self.train_images = (self.train_images - mean) / std
        self.val_images = (self.val_images - mean) / std
        self.test_images = (self.test_images - mean) / std

        assert len(self.test_images) == len(self.test_labels)
        assert len(self.train_images) == len(self.train_labels)
        assert len(self.val_images) == len(self.val_labels)
        tf.logging.info('train dataset size: {}, test: {}, val: {}'.format(
            len(self.train_images), len(self.test_images), len(self.val_images)))

    def parse_policy(self, hparams):
        """Parses policy schedule from input, which can be a list, list of lists, text file, or pickled list.

        If list is not nested, then uses the same policy for all epochs.

        Args:
        hparams: tf.hparams object.
        """
        # Parse policy
        self.augmentation_transforms = augmentation_transforms_pba

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
        elif isinstance(hparams.hp_policy,
                        str) and hparams.hp_policy.endswith('.p'):
            assert hparams.num_epochs % hparams.hp_policy_epochs == 0
            tf.logging.info('custom .p file, policy number: {}'.format(
                hparams.schedule_num))
            with open(hparams.hp_policy, 'rb') as f:
                policy = pickle.load(f)[hparams.schedule_num]
            raw_policy = []
            for num_iters, pol in policy:
                for _ in range(num_iters * hparams.num_epochs //
                               hparams.hp_policy_epochs):
                    raw_policy.append(pol)
        else:
            raw_policy = hparams.hp_policy

        if isinstance(raw_policy[0], list):
            self.policy = []
            split = len(raw_policy[0]) // 2
            for pol in raw_policy:
                cur_pol = parse_policy(pol[:split],
                                       self.augmentation_transforms)
                cur_pol.extend(
                    parse_policy(pol[split:],
                                 self.augmentation_transforms))
                self.policy.append(cur_pol)
            tf.logging.info('using HP policy schedule, last: {}'.format(
                self.policy[-1]))
        elif isinstance(raw_policy, list):
            split = len(raw_policy) // 2
            self.policy = parse_policy(raw_policy[:split],
                                       self.augmentation_transforms)
            self.policy.extend(
                parse_policy(raw_policy[split:],
                             self.augmentation_transforms))
            tf.logging.info('using HP Policy, policy: {}'.format(
                self.policy))

    def reset_policy(self, new_hparams):
        self.hparams = new_hparams
        self.parse_policy(new_hparams)
        tf.logging.info('reset aug policy')
        return

    def load_data(self):
        """Load raw data from hack dataset.

        Assumes data is in NCHW format.

        Populates:
            self.train_images: Training image data.
            self.train_labels: Training ground truth labels.
            self.val_images: Validation/holdout image data.
            self.val_labels: Validation/holdout ground truth labels.
            self.test_images: Testing image data.
            self.test_labels: Testing ground truth labels.
            self.num_classes: Number of classes.
            self.num_train: Number of training examples.
            self.image_size: Width/height of image.

        Args:
            hparams: tf.hparams object.
        """
        train_data_root = '/data/zwy/datasetv4/align/train'
        train_csv_path = '/data/zwy/datasetv4/align/datasetv5_train.csv'
        val_data_root = '/data/zwy/datasetv4/align/train'
        val_csv_path = '/data/zwy/datasetv4/align/jdb.csv'
        test_data_root = '/data/zwy/datasetv4/align/train'
        test_csv_path = '/data/zwy/datasetv4/align/jdb.csv'
        import torchvision.transforms.functional as TF
        import torchvision.transforms as transforms
        crop = transforms.Lambda(lambda img: TF.crop(img, 251 - 250, 273 - 250, 500, 500))
        transform = transforms.Compose(
            [
                crop,
                transforms.Resize(224),
            ]
        )
        self.train_loader = DataLoader(CsvDataset(train_data_root, train_csv_path, transform=transform),
                                       batch_size=self.hparams.batch_size,
                                       shuffle=True,
                                       num_workers=2)
        self.val_loader = DataLoader(CsvDataset(val_data_root, val_csv_path, transform=transform),
                                     batch_size=self.hparams.batch_size,
                                     shuffle=False,
                                     num_workers=2)
        self.test_loader = DataLoader(CsvDataset(test_data_root, test_csv_path, transform=transform),
                                      batch_size=self.hparams.test_batch_size,
                                      shuffle=False,
                                      num_workers=2)
        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader)
        self.test_iter = iter(self.test_loader)

        self.num_classes = 2
        self.num_train = len(self.train_dataset)
        self.num_val = len(self.val_dataset)
        self.num_test = len(self.test_dataset)
        self.image_size = 224

    def next_batch(self, iteration=None):
        """Return the next minibatch of augmented data."""
        try:
            batched_data = next(self.train_iter)
        except StopIteration:
            # Increase epoch number
            epoch = self.epochs + 1
            self.reset()
            self.epochs = epoch
            batched_data = next(self.train_iter)
        final_imgs = []

        images, labels = batched_data
        for data in images:
            if not self.hparams.no_aug:
                # apply PBA policy)
                if isinstance(self.policy[0], list):
                    final_img = self.augmentation_transforms.apply_policy(
                        self.policy[iteration],
                        data,
                        image_size=self.image_size)
                elif isinstance(self.policy, list):
                    # policy schedule
                    final_img = self.augmentation_transforms.apply_policy(
                        self.policy,
                        data,
                        image_size=self.image_size)
                else:
                    raise ValueError('Unknown policy.')
            else:
                final_img = data
            final_img = self.augmentation_transforms.random_flip(
                self.augmentation_transforms.zero_pad_and_crop(
                    final_img, 16))

            # Apply cutout
            if not self.hparams.no_cutout:
                final_img = self.augmentation_transforms.cutout_numpy(
                    final_img, size=56)
            final_imgs.append(final_img)
        batched_data = (np.array(final_imgs, np.float32), labels)
        return batched_data

    def reset(self):
        """Reset training data and index into the training data."""
        self.epochs = 0
        # Shuffle the training data
        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader)
        self.test_iter = iter(self.test_loader)
