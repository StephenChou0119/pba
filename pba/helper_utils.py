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

from autoaugment.helper_utils import setup_loss, decay_weights, cosine_lr  # pylint: disable=unused-import

def step_lr(learning_rate, epoch):
    """Step Learning rate.

  Args:
    learning_rate: Initial learning rate.
    epoch: Current epoch we are one. This is one based.

  Returns:
    The learning rate to be used for this current batch.
  """
    if epoch < 80:
        return learning_rate
    elif epoch < 120:
        return learning_rate * 0.1
    else:
        return learning_rate * 0.01


def get_lr(curr_epoch, hparams, iteration=None):
    """Returns the learning rate during training based on the current epoch."""
    assert iteration is not None
    batches_per_epoch = int(hparams.train_size / hparams.batch_size)

    lr = cosine_lr(hparams.lr, curr_epoch, iteration, batches_per_epoch,
                   hparams.num_epochs)

    return lr
