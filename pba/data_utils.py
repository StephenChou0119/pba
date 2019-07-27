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

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from pba.cutout import Cutout
import torchvision

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
        Args:
            hparams: tf.hparams object.
            todo write doc of hparams
    """

    def __init__(self, hparams):
        self.__cutout_size = hparams.cutout_size
        self.__num_workers = hparams.num_workers
        self.__padding_size = hparams.padding_size
        self.image_size = hparams.size_of_image
        self.num_classes = hparams.num_of_classes
        self.hparams = hparams
        self.augmentation_transforms = augmentation_transforms_pba

        self.parse_policy(hparams)
        self.__pba_transform = transforms.Lambda(lambda img: self.__apply_pba(img))
        self.__normalize = transforms.Normalize(hparams.mean, hparams.std)
        self.__test_transform = transforms.Compose(
            [
                # transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                # self.__normalize,
            ]
        )
        self.reset_dataloader()
        self.__curr_epoch = 0

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
            tf.logging.info('using cifar10 config!')
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
            tf.logging.info('using svhn config!')
            self.data_root = self.hparams.data_root
            self.__transform = transforms.Compose(
                [
                    # transforms.Resize((self.image_size, self.image_size)),
                    # self.__pba_transform,
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomCrop(size=(self.image_size, self.image_size), padding=self.__padding_size),
                    transforms.ToTensor(),
                    # self.__normalize,
                    # Cutout(n_holes=1, length=self.__cutout_size)
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

    def __apply_pba(self, data):
        """

        Args:
            data: pillow image object

        Returns:
            pillow image object
        """
        # apply PBA policy
        if isinstance(self.policy[0], list):
            img = self.augmentation_transforms.apply_policy(
                self.policy[self.__curr_epoch],
                data,
                image_size=self.image_size)
        elif isinstance(self.policy, list):
            # policy schedule
            img = self.augmentation_transforms.apply_policy(
                self.policy,
                data,
                image_size=self.image_size)
        else:
            raise ValueError('Unknown policy.')
        return img

    def parse_policy(self, hparams):
        """Parses policy schedule from input, which can be a list, list of lists, text file, or pickled list.

        If list is not nested, then uses the same policy for all epochs.

        Args:
        hparams: tf.hparams object.
        """
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
        # 2. channel first to channel last
        images, labels = batched_data
        images = images.numpy()
        # images = images.transpose(0,2,3,1)
        labels = labels.numpy()
        batchsize = labels.size
        # 3. label to one hot
        labels = np.eye(self.num_classes)[labels.reshape(-1)].T.reshape(batchsize, -1)
        batched_data = (np.array(images, np.float32), labels)
        return batched_data

