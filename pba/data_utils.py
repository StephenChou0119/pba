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
import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader
from pba.utils import parse_log_schedule
import pba.augmentation_transforms_hp as augmentation_transforms_pba

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
    Load raw data from hack dataset.

        Assumes data is in NHWC format.
    """

    def __init__(self, hparams):
        self.__train_data_root = hparams.train_data_root
        self.__train_csv_path = hparams.train_csv_path
        self.__val_data_root = hparams.val_data_root
        self.__val_csv_path = hparams.val_csv_path
        self.__test_data_root = hparams.test_data_root
        self.__test_csv_path = hparams.test_csv_path

        self.__transform = hparams.transform
        self.__cutout_size = hparams.cutout_size
        self.__padding_size = hparams.padding_size
        self.image_size = hparams.size_of_image
        self.num_classes = hparams.num_of_classes
        self.hparams = hparams
        self.epochs = 0

        self.parse_policy(hparams)
        self.train_loader = DataLoader(
            CsvDataset(self.__train_data_root, self.__train_csv_path, transform=self.__transform),
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True)
        self.val_loader = DataLoader(CsvDataset(self.__val_data_root, self.__val_csv_path, transform=self.__transform),
                                     batch_size=self.hparams.test_batch_size,
                                     shuffle=False,
                                     num_workers=2,
                                     drop_last=True)
        self.test_loader = DataLoader(
            CsvDataset(self.__test_data_root, self.__test_csv_path, transform=self.__transform),
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            num_workers=2,
            drop_last=True)
        self.__train_iter = iter(self.train_loader)

        self.num_train = len(self.train_loader)
        self.num_val = len(self.val_loader)
        self.num_test = len(self.test_loader)

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
        """Loadrawdatafromhackdataset.

        Assumes data is in NHWC format.

        Populates:
        self.train_loader:Training image and label data.
        self.val_loader:Validation/holdout image and label data.
        self.test_loader:Testing image and label data.
        self.num_classes:Number of classes.
        self.num_train:Number of training examples.
        self.num_val:Number of validating examples.
        self.num_test:Number of testing examples.
        self.image_size:Width/height of image.

        Args:
        hparams:tf.hparams object.
        """

        self.train_loader = DataLoader(CsvDataset(self.__train_data_root, self.__train_csv_path, transform=self.__transform),
                                       batch_size=self.hparams.batch_size,
                                       shuffle=True,
                                       num_workers=2,
                                       drop_last=True)
        self.val_loader = DataLoader(CsvDataset(self.__val_data_root, self.__val_csv_path, transform=self.__transform),
                                     batch_size=self.hparams.test_batch_size,
                                     shuffle=False,
                                     num_workers=2,
                                     drop_last=True)
        self.test_loader = DataLoader(CsvDataset(self.__test_data_root, self.__test_csv_path, transform=self.__transform),
                                      batch_size=self.hparams.test_batch_size,
                                      shuffle=False,
                                      num_workers=2,
                                      drop_last=True)
        self.__train_iter = iter(self.train_loader)

        self.num_train = len(self.train_loader)
        self.num_val = len(self.val_loader)
        self.num_test = len(self.test_loader)

    def next_batch(self, iteration=None):
        """Return the next minibatch of augmented data."""
        try:
            batched_data = next(self.__train_iter)
        except StopIteration:
            # Increase epoch number
            epoch = self.epochs + 1
            self.reset()
            self.epochs = epoch
            batched_data = next(self.__train_iter)
        final_imgs = []

        images, labels = batched_data
        images = images.numpy()
        images = images.transpose(0,2,3,1)
        labels = labels.numpy()
        batchsize = labels.size
        labels = np.eye(2)[labels.reshape(-1)].T.reshape(batchsize, -1)

        for data in images:
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
            final_img = self.augmentation_transforms.random_flip(
                self.augmentation_transforms.zero_pad_and_crop(
                    final_img, self.__padding_size))

            # Apply cutout
            if not self.hparams.no_cutout:
                final_img = self.augmentation_transforms.cutout_numpy(
                    final_img, size=self.__cutout_size)
            final_imgs.append(final_img)
        batched_data = (np.array(final_imgs, np.float32), labels)
        return batched_data

    def reset(self):
        """Reset training data and index into the training data."""
        self.epochs = 0
        # Shuffle the training data
        self.__train_iter = iter(self.train_loader)

