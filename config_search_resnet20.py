from models.efficientnet_builder import build_model as build_efficientnet
from models.resnet import build_resnet_model
local_dir = "/data/search_results/"  # Ray directory.
restore = None  # if not None, tries to restore from given path.
checkpoint_freq = 50
cpu = 40  # cpu allocated by Ray for each trial
gpu = 8  # gpu allocated by Ray for each trial
epochs = 100  # number of epochs, must > 0
learning_rate = 0.1
weight_decay = 0.0005
batch_size = 64
test_batch_size = 64
num_samples = 16  # number of trials

# name of directory created in local_dir to store search temp files, checkpoints, policies.
search_name = 'resnet20_search'
train_name = 'resnet20_train'

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
    logits = build_resnet_model(inputs, num_classes, 20, is_training, )
    return logits
