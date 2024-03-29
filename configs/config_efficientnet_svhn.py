local_dir = "/home/sun/zwy/pba_results/"  # Ray directory.
restore = None  # if not None, tries to restore from given path.
checkpoint_freq = 50
cpu = 5  # cpu allocated by Ray for each trial
gpu = 1  # gpu allocated by Ray for each trial
epochs = 1800  # number of epochs, must > 0
learning_rate = 0.01
weight_decay = 0.0005
batch_size = 4096
test_batch_size = 4096

# name of directory created in local_dir to store search temp files, checkpoints, policies.
search_name = 'efficientnet_search_svhn'
train_name = 'efficientnet_train_svhn_nopba'

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


# arguments for search
num_samples = 16  # number of trials
perturbation_interval = 3  # interval for moving top 25% to last 25%

# arguments for train
# hp_policy = '/home/sun/zwy/pba_tensorflow/schedules/svhn200_efficientnet.txt'
hp_policy = None
hp_policy_epochs = 200  # epochs should be able to divide evenly into hp_policy_epochs

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

# metric and mode
metric = 'val_loss'
mode = 'min'
