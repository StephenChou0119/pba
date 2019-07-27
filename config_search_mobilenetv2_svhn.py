import tensorflow as tf

local_dir = "/home/sun/zwy/search_results/"  # Ray directory.
restore = None  # if not None, tries to restore from given path.
checkpoint_freq = 50
cpu = 5  # cpu allocated by Ray for each trial
gpu = 1  # gpu allocated by Ray for each trial
epochs = 200  # number of epochs, must > 0
learning_rate = 0.1
weight_decay = 0.0005
batch_size = 4096
test_batch_size = 4096
# num_samples = 16  # number of trials
num_samples = 1
# name of directory created in local_dir to store search temp files, checkpoints, policies.
search_name = 'mobilenetv2_search_svhn_test'
train_name = 'mobilenetv2_train_svhn_test'


# arguments for train
# hp_policy = '/data/zwy/hack_pba_tensorflow/schedules/cifar10.txt'
hp_policy = ''
hp_policy_epochs = 200  # epochs should be able to divide evenly into hp_policy_epochs

# interval for moving top 25% to last 25%
perturbation_interval = 4

# dataset
dataset_type = 'svhn'
data_root = '/home/sun/zwy/data_svhn'  # path to store dataset
mean = [0.43090966, 0.4302428, 0.44634357]
std = [0.19652855, 0.19832038, 0.19942076]
padding_size = 4
cutout_size = 20
num_workers = 5  # cpu number used for loading data
image_size = 32
num_classes = 10


# model builder
def build_func(inputs, num_classes, is_training):
    import models.mobilenet.mobilenet_v2 as mobilenet_v2
    if is_training:
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
            logits, endpoints = mobilenet_v2.mobilenet_v2_035(inputs, num_classes=num_classes)
    else:
        logits, endpoints = mobilenet_v2.mobilenet_v2_035(inputs, num_classes=num_classes)
    return logits
