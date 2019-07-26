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
"""PBA & AutoAugment Train/Eval module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import time

import numpy as np
import tensorflow as tf

import pba.data_utils as data_utils
import pba.helper_utils as helper_utils
# from pba.setup import build_func

arg_scope = tf.contrib.framework.arg_scope


@tf.contrib.framework.add_arg_scope
def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               is_training=True,
               reuse=None,
               scope=None):
  """Small wrapper around tf.contrib.layers.batch_norm."""
  return tf.contrib.layers.batch_norm(
      inputs,
      decay=decay,
      center=center,
      scale=scale,
      epsilon=epsilon,
      activation_fn=None,
      param_initializers=None,
      updates_collections=tf.GraphKeys.UPDATE_OPS,
      is_training=is_training,
      reuse=reuse,
      trainable=True,
      fused=True,
      data_format='NHWC',
      zero_debias_moving_mean=False,
      scope=scope)


def setup_arg_scopes(is_training):
    """Sets up the argscopes that will be used when building an image model.

    Args:
      is_training: Is the model training or not.

    Returns:
      Arg scopes to be put around the model being constructed.
    """

    batch_norm_decay = 0.9
    batch_norm_epsilon = 1e-5
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': batch_norm_decay,
        # epsilon to prevent 0s in variance.
        'epsilon': batch_norm_epsilon,
        'scale': True,
        # collection containing the moving mean and moving variance.
        'is_training': is_training,
    }

    scopes = []

    scopes.append(arg_scope([batch_norm], **batch_norm_params))
    return scopes


def build_model(inputs, num_classes, is_training, hparams):
    """Constructs the vision model being trained/evaled.

    Args:
      inputs: input features/images being fed to the image model build built.
      num_classes: number of output classes being predicted.
      is_training: is the model training or not.
      hparams: additional hyperparameters associated with the image model.

    Returns:
      The logits of the image model.
    """
    scopes = setup_arg_scopes(is_training)
    if len(scopes) != 1:
        raise ValueError('Nested scopes depreciated in py3.')
    with scopes[0]:
        # logits = build_func(inputs, num_classes, is_training)
        def build_func(inputs, num_classes, is_training):
            import models.mobilenet.mobilenet_v2 as mobilenet_v2
            if is_training:
                with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
                    logits, endpoints = mobilenet_v2.mobilenet_v2_035(inputs, num_classes=num_classes)
            else:
                logits, endpoints = mobilenet_v2.mobilenet_v2_035(inputs, num_classes=num_classes)
            return logits
        logits = build_func(inputs, num_classes, is_training)

    return logits


def run_epoch_training(session, model, dataset, curr_epoch):
    """Runs one epoch of training for the model passed in.

  Args:
    session: TensorFlow session the model will be run with.
    model: TensorFlow model that will be evaluated.
    dataset: DataSet object that contains data that `model` will evaluate.
    curr_epoch: How many of epochs of training have been done so far.

  Returns:
    The accuracy of 'model' on the training set
  """
    steps_per_epoch = int(model.hparams.train_size / model.hparams.batch_size)
    tf.logging.info('steps per epoch: {}'.format(steps_per_epoch))
    curr_step = session.run(model.global_step)
    assert curr_step % steps_per_epoch == 0

    # Get the current learning rate for the model based on the current epoch
    curr_lr = helper_utils.get_lr(curr_epoch, model.hparams, iteration=0)
    tf.logging.info('lr of {} for epoch {}'.format(curr_lr, curr_epoch))

    correct = 0
    count = 0
    for step in range(steps_per_epoch):
        curr_lr = helper_utils.get_lr(curr_epoch, model.hparams, iteration=(step + 1))
        # Update the lr rate variable to the current LR.
        model.lr_rate_ph.load(curr_lr, session=session)
        if step % 20 == 0:
            tf.logging.info('Training {}/{}'.format(step, steps_per_epoch))

        train_images, train_labels = dataset.next_batch()

        _, step, preds = session.run(
            [model.train_op, model.global_step, model.predictions],
            feed_dict={
                model.images: train_images,
                model.labels: train_labels,
            })

        correct += np.sum(
            np.equal(np.argmax(train_labels, 1), np.argmax(preds, 1)))
        count += len(preds)
    train_accuracy = correct / count

    tf.logging.info('Train accuracy: {}'.format(train_accuracy))
    return train_accuracy


def eval_child_model(session, model, dataset, mode):
    """Evaluates `model` on held out data depending on `mode`.

  Args:
    session: TensorFlow session the model will be run with.
    model: TensorFlow model that will be evaluated.
    dataset: DataSet object that contains data that `model` will evaluate.
    mode: Will `model` either evaluate validation or test data.

  Returns:
    Accuracy of `model` when evaluated on the specified dataset.

  Raises:
    ValueError: if invalid dataset `mode` is specified.
  """
    if mode == 'val':
        loader = dataset.val_loader
    elif mode == 'test':
        loader = dataset.test_loader
    else:
        raise ValueError('Not valid eval mode')
    tf.logging.info('model.batch_size is {}'.format(model.batch_size))

    correct = 0
    count = 0
    for i, batch in enumerate(loader):
        images, labels = batch
        images = images.numpy()
        images = images.transpose(0, 2, 3, 1)
        labels = labels.numpy()
        batchsize = labels.size
        labels = np.eye(dataset.num_classes)[labels.reshape(-1)].T.reshape(batchsize, -1)
        preds = session.run(
            model.predictions,
            feed_dict={
                model.images: images,
                model.labels: labels,
            })
        correct += np.sum(
            np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))
        count += len(preds)
    return correct / count


class Model(object):
    """Builds an model."""

    def __init__(self, hparams, num_classes, image_size):
        self.hparams = hparams
        self.num_classes = num_classes
        self.image_size = image_size

    def build(self, mode):
        """Construct the model."""
        assert mode in ['train', 'eval']
        self.mode = mode
        self._setup_misc(mode)
        self._setup_images_and_labels()
        self._build_graph(self.images, self.labels, mode)

        self.init = tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer())

    def _setup_misc(self, mode):
        """Sets up miscellaneous in the model constructor."""
        self.lr_rate_ph = tf.Variable(0.0, name='lrn_rate', trainable=False)
        self.reuse = None if (mode == 'train') else True
        self.batch_size = self.hparams.batch_size

    def _setup_images_and_labels(self):
        """Sets up image and label placeholders for the model."""
        self.images = tf.placeholder(tf.float32,
                                     [self.batch_size, self.image_size, self.image_size, 3])
        self.labels = tf.placeholder(tf.float32,
                                     [self.batch_size, self.num_classes])

    def assign_epoch(self, session, epoch_value):
        session.run(
            self._epoch_update, feed_dict={self._new_epoch: epoch_value})

    def _build_graph(self, images, labels, mode):
        """Constructs the TF graph for the model.

        Args:
          images: A 4-D image Tensor
          labels: A 2-D labels Tensor.
          mode: string indicating training mode ( e.g., 'train', 'valid', 'test').
        """
        is_training = 'train' in mode
        if is_training:
            self.global_step = tf.train.get_or_create_global_step()

        logits = build_model(images, self.num_classes, is_training,
                             self.hparams)
        self.predictions, self.cost = helper_utils.setup_loss(logits, labels)

        self._calc_num_trainable_params()

        # Adds L2 weight decay to the cost
        self.cost = helper_utils.decay_weights(self.cost,
                                               self.hparams.weight_decay_rate)

        if is_training:
            self._build_train_op()

        # Setup checkpointing for this child model
        # Keep 2 or more checkpoints around during training.
        with tf.device('/cpu:0'):
            self.saver = tf.train.Saver(max_to_keep=10)

        self.init = tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer())

    def _calc_num_trainable_params(self):
        self.num_trainable_params = np.sum([
            np.prod(var.get_shape().as_list())
            for var in tf.trainable_variables()
        ])
        tf.logging.info('number of trainable params: {}'.format(
            self.num_trainable_params))

    def _build_train_op(self):
        """Builds the train op for the model."""
        hparams = self.hparams
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.cost, tvars)
        if hparams.gradient_clipping_by_global_norm > 0.0:
            grads, norm = tf.clip_by_global_norm(
                grads, hparams.gradient_clipping_by_global_norm)
            tf.summary.scalar('grad_norm', norm)

        # Setup the initial learning rate
        initial_lr = self.lr_rate_ph
        optimizer = tf.train.MomentumOptimizer(
            initial_lr, 0.9, use_nesterov=True)

        self.optimizer = optimizer
        apply_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=self.global_step, name='train_step')
        train_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([apply_op]):
            self.train_op = tf.group(*train_ops)


class ModelTrainer(object):
    """Trains an instance of the Model class."""

    def __init__(self, hparams):
        self._session = None
        self.hparams = hparams

        # Set the random seed to be sure the same validation set
        # is used for each model
        np.random.seed(0)
        self.dataset = data_utils.DataSet(hparams)
        np.random.seed()  # Put the random seed back to random
        # extra stuff for ray
        self._build_models()
        self._new_session()
        self._session.__enter__()

    def save_model(self, checkpoint_dir, step=None):
        """Dumps model into the backup_dir.

        Args:
          step: If provided, creates a checkpoint with the given step
            number, instead of overwriting the existing checkpoints.
        """
        model_save_name = os.path.join(checkpoint_dir,
                                       'model.ckpt') + '-' + str(step)
        save_path = self.saver.save(self.session, model_save_name)
        tf.logging.info('Saved child model')
        return model_save_name

    def restore(self, checkpoint_path):
        """Loads a checkpoint with the architecture structure stored in the name."""
        self.saver.restore(self.session, checkpoint_path)
        tf.logging.warning(
            'Loaded child model checkpoint from {}'.format(checkpoint_path))

    def eval_child_model(self, model, dataset, mode):
        """Evaluate the child model.

        Args:
          model: image model that will be evaluated.
          dataset: dataset object to extract eval data from.
          mode: will the model be evalled on train, val or test.

        Returns:
          Accuracy of the model on the specified dataset.
        """
        tf.logging.info('Evaluating child model in mode {}'.format(mode))
        while True:
            try:
                accuracy = eval_child_model(
                    self.session, model, dataset, mode)
                tf.logging.info(
                    'Eval child model accuracy: {}'.format(accuracy))
                # If epoch trained without raising the below errors, break
                # from loop.
                break
            except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
                tf.logging.info(
                    'Retryable error caught: {}.  Retrying.'.format(e))

        return accuracy

    @contextlib.contextmanager
    def _new_session(self):
        """Creates a new session for model m."""
        # Create a new session for this model, initialize
        # variables, and save / restore from checkpoint.
        sess_cfg = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        sess_cfg.gpu_options.allow_growth = True
        self._session = tf.Session('', config=sess_cfg)
        self._session.run([self.m.init, self.meval.init])
        return self._session

    def _build_models(self):
        """Builds the image autoaugment for train and eval."""
        # Determine if we should build the train and eval model. When using
        # distributed training we only want to build one or the other and not both.
        with tf.variable_scope('model', use_resource=False):
            m = Model(self.hparams, self.dataset.num_classes, self.dataset.image_size)
            m.build('train')
            self._num_trainable_params = m.num_trainable_params
            self._saver = m.saver
        with tf.variable_scope('model', reuse=True, use_resource=False):
            meval = Model(self.hparams, self.dataset.num_classes, self.dataset.image_size)
            meval.build('eval')
        self.m = m
        self.meval = meval
        self.m.hparams.add_hparam('train_size', len(self.dataset.train_loader.dataset))

    def _run_training_loop(self, curr_epoch):
        """Trains the model `m` for one epoch."""
        start_time = time.time()
        while True:
            try:
                train_accuracy = run_epoch_training(
                    self.session, self.m, self.dataset, curr_epoch)
                break
            except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
                tf.logging.info(
                    'Retryable error caught: {}.  Retrying.'.format(e))
        tf.logging.info('Finished epoch: {}'.format(curr_epoch))
        tf.logging.info('Epoch time(min): {}'.format(
            (time.time() - start_time) / 60.0))
        return train_accuracy

    def _compute_final_accuracies(self, iteration):
        """Run once training is finished to compute final test accuracy."""
        if (iteration >= self.hparams.num_epochs - 1):
            test_accuracy = self.eval_child_model(self.meval, self.dataset,
                                                  'test')
        else:
            test_accuracy = 0
        tf.logging.info('Test Accuracy: {}'.format(test_accuracy))
        return test_accuracy

    def run_model(self, curr_epoch):
        """Trains and evalutes the image model."""
        # set curr_epoch for load data and apply pba
        self.dataset.curr_epoch = curr_epoch
        self.dataset.reset_dataloader()
        training_accuracy = self._run_training_loop(curr_epoch)
        valid_accuracy = self.eval_child_model(self.meval,
                                                   self.dataset, 'val')
        tf.logging.info('Train Acc: {}, Valid Acc: {}'.format(
            training_accuracy, valid_accuracy))
        return training_accuracy, valid_accuracy

    def reset_config(self, new_hparams):
        self.hparams = new_hparams
        self.dataset.reset_policy(new_hparams)
        return

    @property
    def saver(self):
        return self._saver

    @property
    def session(self):
        return self._session

    @property
    def num_trainable_params(self):
        return self._num_trainable_params
