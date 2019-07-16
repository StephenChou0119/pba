from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Parse flags and set up hyperparameters."""
import argparse
import random
import tensorflow as tf

from pba.augmentation_transforms_hp import NUM_HP_TRANSFORM


def create_parser(state):
    """Create arg parser for flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        default='wrn_28_10',
        choices=('wrn_28_10', 'wrn_40_2', 'shake_shake_32', 'shake_shake_96',
                 'shake_shake_112', 'pyramid_net', 'resnet'))
    parser.add_argument('--local_dir', type=str, default='/tmp/ray_results/',  help='Ray directory.')
    parser.add_argument('--restore', type=str, default=None, help='If specified, tries to restore from given path.')
    parser.add_argument('--checkpoint_freq', type=int, default=50, help='Checkpoint frequency.')
    parser.add_argument(
        '--cpu', type=float, default=4, help='Allocated by Ray')
    parser.add_argument(
        '--gpu', type=float, default=1, help='Allocated by Ray')
    # search-use only
    parser.add_argument(
        '--explore',
        type=str,
        default='cifar10',
        help='which explore function to use')
    parser.add_argument(
        '--epochs',
        type=int,
        default=0,
        help='Number of epochs, or <=0 for default')
    parser.add_argument(
        '--no_cutout', action='store_true', help='turn off cutout')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--bs', type=int, default=512, help='batch size')
    parser.add_argument('--test_bs', type=int, default=512, help='test batch size')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of Ray samples')
    parser.add_argument('--resnet_size', type=int, default=50, help='Number of Ray samples')

    if state == 'train':
        parser.add_argument(
            '--use_hp_policy',
            action='store_true',
            help='otherwise use autoaug policy')
        parser.add_argument(
            '--hp_policy',
            type=str,
            default=None,
            help='either a comma separated list of values or a file')
        parser.add_argument(
            '--hp_policy_epochs',
            type=int,
            default=100,
            help='number of epochs/iterations policy trained for')
        parser.add_argument(
            '--no_aug',
            action='store_true',
            help=
            'no additional augmentation at all (besides cutout if not toggled)'
        )
        parser.add_argument('--name', type=str, default='autoaug')

    elif state == 'search':
        parser.add_argument('--perturbation_interval', type=int, default=3)
        parser.add_argument('--name', type=str, default='autoaug_pbt')
    else:
        raise ValueError('unknown state')
    args = parser.parse_args()
    tf.logging.info(str(args))
    return args


def create_hparams(state, FLAGS):  # pylint: disable=invalid-name
    """Creates hyperparameters to pass into Ray config.

  Different options depending on search or eval mode.

  Args:
    state: a string, 'train' or 'search'.
    FLAGS: parsed command line flags.

  Returns:
    tf.hparams object.
  """
    epochs = 0
    hparams = tf.contrib.training.HParams(
        batch_size=FLAGS.bs,
        gradient_clipping_by_global_norm=5.0,
        explore=FLAGS.explore,
        no_cutout=FLAGS.no_cutout,
        lr=FLAGS.lr,
        weight_decay_rate=FLAGS.wd,
        test_batch_size=FLAGS.test_bs)

    if state == 'train':
        hparams.add_hparam('no_aug', FLAGS.no_aug)
        hparams.add_hparam('use_hp_policy', FLAGS.use_hp_policy)
        if FLAGS.use_hp_policy:
            if FLAGS.hp_policy == 'random':
                tf.logging.info('RANDOM SEARCH')
                parsed_policy = []
                for i in range(NUM_HP_TRANSFORM * 4):
                    if i % 2 == 0:
                        parsed_policy.append(random.randint(0, 10))
                    else:
                        parsed_policy.append(random.randint(0, 9))
            elif FLAGS.hp_policy.endswith('.txt') or FLAGS.hp_policy.endswith(
                    '.p'):
                # will be loaded in in data_utils
                parsed_policy = FLAGS.hp_policy
            else:
                # parse input into a fixed augmentation policy
                parsed_policy = FLAGS.hp_policy.split(', ')
                parsed_policy = [int(p) for p in parsed_policy]
            hparams.add_hparam('hp_policy', parsed_policy)
            hparams.add_hparam('hp_policy_epochs', FLAGS.hp_policy_epochs)
    elif state == 'search':
        hparams.add_hparam('no_aug', False)
        hparams.add_hparam('use_hp_policy', True)
        # default start value of 0
        hparams.add_hparam('hp_policy',
                           [0 for _ in range(4 * NUM_HP_TRANSFORM)])
    else:
        raise ValueError('unknown state')

    if FLAGS.model_name == 'wrn_40_2':
        hparams.add_hparam('model_name', 'wrn')
        hparams.add_hparam('wrn_size', 32)
        hparams.add_hparam('wrn_depth', 40)
    elif FLAGS.model_name == 'wrn_28_10':
        hparams.add_hparam('model_name', 'wrn')
        hparams.add_hparam('wrn_size', 160)
        hparams.add_hparam('wrn_depth', 28)
    elif FLAGS.model_name == 'resnet':
        hparams.add_hparam('model_name', 'resnet')
        hparams.add_hparam('resnet_size', 20)
        hparams.add_hparam('num_filters', FLAGS.resnet_size)
    elif FLAGS.model_name == 'shake_shake_32':
        hparams.add_hparam('model_name', 'shake_shake')
        hparams.add_hparam('shake_shake_widen_factor', 2)
    elif FLAGS.model_name == 'shake_shake_96':
        hparams.add_hparam('model_name', 'shake_shake')
        hparams.add_hparam('shake_shake_widen_factor', 6)
    elif FLAGS.model_name == 'shake_shake_112':
        hparams.add_hparam('model_name', 'shake_shake')
        hparams.add_hparam('shake_shake_widen_factor', 7)
    elif FLAGS.model_name == 'pyramid_net':
        hparams.add_hparam('model_name', 'pyramid_net')
        hparams.set_hparam('batch_size', 64)
    else:
        raise ValueError('Not Valid Model Name: %s' % FLAGS.model_name)
    if FLAGS.epochs > 0:
        epochs = FLAGS.epochs
    else:
        raise ValueError('epochs must larger than 0')
    hparams.add_hparam('num_epochs', epochs)
    tf.logging.info('epochs: {}, lr: {}, wd: {}'.format(
        hparams.num_epochs, hparams.lr, hparams.weight_decay_rate))
    return hparams
