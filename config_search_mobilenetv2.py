import tensorflow as tf
import models.mobilenet.mobilenet_v2 as mobilenet_v2

local_dir = "/data/search_results/"  # Ray directory.
restore = None  # if not None, tries to restore from given path.
checkpoint_freq = 50
cpu = 5  # cpu allocated by Ray for each trial
gpu = 1  # gpu allocated by Ray for each trial
epochs = 100  # number of epochs, must > 0
learning_rate = 0.1
weight_decay = 0.0005
batch_size = 256
test_batch_size = 256
num_samples = 4  # number of trials

# name of directory created in local_dir to store search temp files, checkpoints, policies.
search_name = 'mobilenetv2_search_test'
train_name = 'mobilenetv2_train_test'

# cifar crop padding size, cutout size
padding_size = 16
cutout_size = 56
num_workers = 5
no_cutout = False
# arguments for train
hp_policy = '/data/zwy/hack_pba_tensorflow/schedules/jdb.txt'
hp_policy_epochs = 100  # epochs should be able to divide evenly into hp_policy_epochs

# interval for moving top 25% to last 25%
perturbation_interval = 1

# dataset

# train
train_data_root = '/data/zwy/datasetv4/align/train'
train_csv_path = '/data/zwy/datasetv4/align/datasetv5_train.csv'
val_data_root = '/data/zwy/datasetv4/align/train'
val_csv_path = '/data/zwy/datasetv4/align/search_val_jdb.csv'
test_data_root = '/data/zwy/datasetv4/align/train'
test_csv_path = '/data/zwy/datasetv4/align/search_test_jdb.csv'

# debug
# train_data_root = '/data/zwy/datasetv4/align/train'
# train_csv_path = '/data/zwy/hack_pba_tensorflow/train100.csv'
# val_data_root = '/data/zwy/datasetv4/align/train'
# val_csv_path = '/data/zwy/hack_pba_tensorflow/val30.csv'
# test_data_root = '/data/zwy/datasetv4/align/train'
# test_csv_path = '/data/zwy/hack_pba_tensorflow/test30.csv'

# search
# train_data_root = '/data/zwy/datasetv4/align/train'
# train_csv_path = '/data/zwy/datasetv4/align/search.csv'
# val_data_root = '/data/zwy/datasetv4/align/train'
# val_csv_path = '/data/zwy/datasetv4/align/search_val_jdb.csv'
# test_data_root = '/data/zwy/datasetv4/align/train'
# test_csv_path = '/data/zwy/datasetv4/align/search_test_jdb.csv'
# preprocess
image_size = 224
num_classes = 2

# model builder


def build_func(inputs, num_classes, is_training):
    if is_training:
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
            logits, endpoints = mobilenet_v2.mobilenet_v2_035(inputs, num_classes=num_classes)
    else:
        logits, endpoints = mobilenet_v2.mobilenet_v2_035(inputs, num_classes=num_classes)
    return logits
