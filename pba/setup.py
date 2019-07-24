from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Parse flags and set up hyperparameters."""
import argparse
import tensorflow as tf

from pba.augmentation_transforms_hp import NUM_HP_TRANSFORM
from mmcv import Config
from ray import tune

def create_parser():
    """Create arg parser for flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')
    args = parser.parse_args()
    config = Config.fromfile(args.config)
    return config


def create_hparams(state, configs):  # pylint: disable=invalid-name
    """Creates hyperparameters to pass into Ray config.

  Different options depending on search or eval mode.

  Args:
    state: a string, 'train' or 'search'.
    configs: parsed command line flags.

  Returns:
    tf.hparams object.
  """
    print(configs.build_func)
    print(tune.function(configs.build_func))
    hparams = tf.contrib.training.HParams(
        batch_size=configs.batch_size,
        gradient_clipping_by_global_norm=5.0,
        lr=configs.learning_rate,
        weight_decay_rate=configs.weight_decay,
        test_batch_size=configs.test_batch_size,
        no_cutout=configs.no_cutout,
        cutout_size=configs.cutout_size,
        padding_size=configs.padding_size,
        size_of_image=configs.image_size,
        num_of_classes=configs.num_classes, # Hyperparameter name is reserved: num_classes, image_size
        build_model=tune.function(configs.build_func),
    )

    if state == 'train':
        if configs.hp_policy.endswith('.txt'):
            # will be loaded in in data_utils
            parsed_policy = configs.hp_policy
        else:
            raise ValueError('policy file must end with .txt')
        hparams.add_hparam('hp_policy', parsed_policy)
        hparams.add_hparam('hp_policy_epochs', configs.hp_policy_epochs)
    elif state == 'search':
        # default start value of 0
        hparams.add_hparam('hp_policy',
                           [0 for _ in range(4 * NUM_HP_TRANSFORM)])
    else:
        raise ValueError('unknown state')

    hparams.add_hparam('train_data_root', configs.train_data_root)
    hparams.add_hparam('train_csv_path', configs.train_csv_path)
    hparams.add_hparam('val_data_root', configs.val_data_root)
    hparams.add_hparam('val_csv_path', configs.val_csv_path)
    hparams.add_hparam('test_data_root', configs.test_data_root)
    hparams.add_hparam('test_csv_path', configs.test_csv_path)
    hparams.add_hparam('image_size', configs.image_size)
    hparams.add_hparam('num_classes', configs.num_classes)
    hparams.add_hparam('num_workers', configs.num_workers)

    if configs.epochs > 0:
        epochs = configs.epochs
    else:
        raise ValueError('epochs must larger than 0')
    hparams.add_hparam('num_epochs', epochs)
    tf.logging.info('epochs: {}, lr: {}, wd: {}'.format(
        hparams.num_epochs, hparams.lr, hparams.weight_decay_rate))
    return hparams
