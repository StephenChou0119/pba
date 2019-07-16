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
"""Helper functions used for training PBA models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def setup_loss(logits, labels):
    """Returns the cross entropy for the given `logits` and `labels`."""
    predictions = tf.nn.softmax(logits)
    cost = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                           logits=logits)
    return predictions, cost


def decay_weights(cost, weight_decay_rate):
    """Calculates the loss for l2 weight decay and adds it to `cost`."""
    costs = []
    for var in tf.trainable_variables():
        costs.append(tf.nn.l2_loss(var))
    cost += tf.multiply(weight_decay_rate, tf.add_n(costs))
    return cost


def get_lr(curr_epoch, hparams, iteration=None):
    """Returns the learning rate during training based on the current epoch."""
    assert iteration is not None
    batches_per_epoch = int(hparams.train_size / hparams.batch_size)

    lr = cosine_lr(hparams.lr, curr_epoch, iteration, batches_per_epoch,
                   hparams.num_epochs)

    return lr


def cosine_lr(learning_rate, epoch, iteration, batches_per_epoch, total_epochs):
    """Cosine Learning rate.

    Args:
      learning_rate: Initial learning rate.
      epoch: Current epoch we are one. This is one based.
      iteration: Current batch in this epoch.
      batches_per_epoch: Batches per epoch.
      total_epochs: Total epochs you are training for.

    Returns:
      The learning rate to be used for this current batch.
    """
    t_total = total_epochs * batches_per_epoch
    t_cur = float(epoch * batches_per_epoch + iteration)
    return 0.5 * learning_rate * (1 + np.cos(np.pi * t_cur / t_total))
