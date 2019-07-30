local_dir = "/data/search_results/"  # Ray directory.
restore = None  # if not None, tries to restore from given path.
checkpoint_freq = 50
cpu = 5  # cpu allocated by Ray for each trial
gpu = 1  # gpu allocated by Ray for each trial
epochs = 100  # number of epochs, must > 0
learning_rate = 0.1
weight_decay = 0.0005
batch_size = 512
test_batch_size = 512

# name of directory created in local_dir to store search temp files, checkpoints, policies.
search_name = 'efficientnet_search_jdb_sensetime'
train_name = 'efficientnet_train_jdb_sensetime_no_aug'

# search space
HP_TRANSFORM_NAMES = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'Posterize',
    'Solarize',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateX',
    'TranslateY',
    'Cutout',
]

rotate_max_degree=30
posterize_max=4
enhance_max=1.8
shear_x_max=0.3
shear_y_max=0.3
translate_x_max=10
translate_y_max=10
cutout_max_size=20

# arguments for search
num_samples = 16  # number of trials
perturbation_interval = 3  # interval for moving top 25% to last 25%

# arguments for train
# hp_policy = '/home/sun/zwy/pba_tensorflow/schedules/svhn200_efficientnet.txt'
hp_policy = None
hp_policy_epochs = 100  # epochs should be able to divide evenly into hp_policy_epochs

# dataset
dataset_type = 'custom'  # choose from 'svhn', 'cifar' or 'custom'
train_data_root = '/data/zwy/datasetv4/align/data'
train_csv_path = '/data/zwy/datasetv4/align/datasetv5_train.csv'
val_data_root = '/data/zwy/datasetv4/align/data'
val_csv_path = '/data/zwy/datasetv4/align/jdb_sensetime_val.csv'
test_data_root = '/data/zwy/datasetv4/align/data'
test_csv_path = '/data/zwy/datasetv4/align/jdb_sensetime_test.csv'
mean = [130.13 / 255, 112.82 / 255, 102.47 / 255]
std = [67.28 / 255, 64.61 / 255, 64.46 / 255]
padding_size = 16
cutout_size = 56
num_workers = 5  # number of cpu used for loading data
image_size = 224
num_classes = 2

# metric and mode
metric = 'val_loss'  # choose from 'val_acc' or 'val_loss'
mode = 'min'  # choose from 'min' or 'max'
