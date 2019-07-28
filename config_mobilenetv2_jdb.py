import tensorflow as tf
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
num_samples = 16  # number of trials

# name of directory created in local_dir to store search temp files, checkpoints, policies.
search_name = 'mobilenetv2_search_jdb'
train_name = 'mobilenetv2_train_jdb'

# cifar crop padding size, cutout size
padding_size = 16
cutout_size = 56
num_workers = 5
no_cutout = False
# arguments for train
hp_policy = '/data/zwy/hack_pba_tensorflow/schedules/jdb.txt'
hp_policy_epochs = 100  # epochs should be able to divide evenly into hp_policy_epochs

# interval for moving top 25% to last 25%
perturbation_interval = 3

# dataset config
dataset_type = 'custom'
train_data_root = '/data/zwy/datasetv4/align/train'
train_csv_path = '/data/zwy/datasetv4/align/datasetv5_train.csv'
val_data_root = '/data/zwy/datasetv4/align/train'
val_csv_path = '/data/zwy/datasetv4/align/search_val_jdb.csv'
test_data_root = '/data/zwy/datasetv4/align/train'
test_csv_path = '/data/zwy/datasetv4/align/search_test_jdb.csv'
dataset_mean = [130.13 / 255, 112.82 / 255, 102.47 / 255]
dataset_std = [67.28 / 255, 64.61 / 255, 64.46 / 255]

image_size = 224
num_classes = 2


# model builder
def build_func(inputs, num_classes, is_training):
    import models.mobilenet.mobilenet_v2 as mobilenet_v2
    if is_training:
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
            logits, endpoints = mobilenet_v2.mobilenet_v2_035(inputs, num_classes=num_classes)
    else:
        logits, endpoints = mobilenet_v2.mobilenet_v2_035(inputs, num_classes=num_classes)
    return logits