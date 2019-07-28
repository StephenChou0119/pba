from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Parse flags and set up hyperparameters."""
import argparse
import tensorflow as tf

from mmcv import Config


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
    hparams = tf.contrib.training.HParams(
        batch_size=configs.batch_size,
        gradient_clipping_by_global_norm=5.0,
        lr=configs.learning_rate,
        weight_decay_rate=configs.weight_decay,
        cutout_size=configs.cutout_size,
        size_of_image=configs.image_size,
        num_of_classes=configs.num_classes,  # Hyperparameter name is reserved: num_classes, image_size
        dataset_type=configs.dataset_type,
        num_workers=configs.num_workers,
        mean=configs.mean,
        std=configs.std,
        padding_size=configs.padding_size,
        HP_TRANSFORM_NAMES=configs.HP_TRANSFORM_NAMES,
        NUM_HP_TRANSFORM=len(configs.HP_TRANSFORM_NAMES),
        # build_func=tune.function(configs.build_func),
    )

    if state == 'train':
        if configs.hp_policy is not None:
            if configs.hp_policy.endswith('.txt'):
                hparams.add_hparam('hp_policy', configs.hp_policy)
                hparams.add_hparam('use_pba', True)
            else:
                raise ValueError('policy file must end with .txt')
            hparams.add_hparam('hp_policy_epochs', configs.hp_policy_epochs)
        else:
            hparams.add_hparam('use_pba', False)
            tf.logging.info('disable pba!')
    elif state == 'search':
        # default start value of 0
        hparams.add_hparam('use_pba', True)
        hparams.add_hparam('hp_policy',
                           [0 for _ in range(4 * hparams.NUM_HP_TRANSFORM)])
    else:
        raise ValueError('unknown state')
    if configs.dataset_type == 'custom':
        hparams.add_hparam('train_data_root', configs.train_data_root)
        hparams.add_hparam('train_csv_path', configs.train_csv_path)
        hparams.add_hparam('val_data_root', configs.val_data_root)
        hparams.add_hparam('val_csv_path', configs.val_csv_path)
        hparams.add_hparam('test_data_root', configs.test_data_root)
        hparams.add_hparam('test_csv_path', configs.test_csv_path)

    else:
        hparams.add_hparam('data_root', configs.data_root)
    if configs.epochs > 0:
        epochs = configs.epochs
    else:
        raise ValueError('epochs must larger than 0')
    hparams.add_hparam('num_epochs', epochs)
    tf.logging.info('epochs: {}, lr: {}, wd: {}'.format(
        hparams.num_epochs, hparams.lr, hparams.weight_decay_rate))
    return hparams
