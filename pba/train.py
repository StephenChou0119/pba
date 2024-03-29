from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Train and evaluate autoaugment using augmentation schedules."""


import os
import ray
from ray.tune import run_experiments
from ray.tune import Trainable
import tensorflow as tf

from pba.model import ModelTrainer
from pba.setup import create_hparams
from pba.setup import create_parser


class RayModel(Trainable):
    """A Ray wrapper for Models to run search."""

    def _setup(self, *args):
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.logging.info("calling setup")
        self.hparams = tf.contrib.training.HParams(**self.config)
        self.trainer = ModelTrainer(self.hparams)

    def _train(self):
        """Runs one epoch of training, and returns current epoch accuracies."""
        tf.logging.info("training for iteration: {}".format(self._iteration))
        train_acc, train_loss, val_acc, val_loss = self.trainer.run_model(self._iteration)
        test_acc = self.trainer.get_test_accuracy(self._iteration)  # pylint: disable=protected-access
        return {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc
        }

    def _save(self, checkpoint_dir):
        """Uses tf trainer object to checkpoint."""
        save_name = self.trainer.save_model(checkpoint_dir, self._iteration)
        tf.logging.info("saved model {}".format(save_name))
        os.close(os.open(save_name, os.O_CREAT))
        return save_name

    def _restore(self, checkpoint):
        """Restores model from checkpoint."""
        tf.logging.info("RESTORING: {}".format(checkpoint))
        self.trainer.restore(checkpoint)

    def reset_config(self, new_config):
        """Resets trainer config for fast PBT implementation."""
        self.config = new_config
        self.hparams = tf.contrib.training.HParams(**new_config)
        self.trainer.reset_config(self.hparams)
        return True


def main(_):
    args = create_parser()  # pylint: disable=invalid-name
    hparams = create_hparams("train", args)

    train_spec = {
        "run": RayModel,
        "resources_per_trial": {
            "cpu": args.cpu,
            "gpu": args.gpu
        },
        "stop": {
            "training_iteration": hparams.num_epochs,
        },
        "config": hparams.values(),
        "local_dir": args.local_dir,
        "checkpoint_freq": args.checkpoint_freq,
        "num_samples": 1
    }

    if args.restore:
        train_spec["restore"] = args.restore

    ray.init()
    run_experiments({args.train_name: train_spec})


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
