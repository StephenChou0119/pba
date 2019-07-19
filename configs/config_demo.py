model_name=False
local_dir = "/data/search_results/"  # Ray directory.
restore = None  # if not None, tries to restore from given path.
checkpoint_freq=50
cpu = 5  # cpu allocated by Ray for each trial
gpu = 1  # gpu allocated by Ray for each trial
epochs = 100  # number of epochs, must > 0
learning_rate = 0.1
weight_decay = 0.0005
batch_size = 128
test_batch_size = 128
num_samples = 16  # number of trials

# name of directory created in local_dir to store search temp files, checkpoints, policies.
search_name = 'efficientnetb0_search'
train_name = 'efficientnetb0_train'

# cifar crop padding size, cutout size
padding_size = 16
cutout_size = 56
no_cutout = False
# arguments for train
hp_policy = '../schedules/rcifar10_16_wrn.txt'
hp_policy_epochs = 100  # epochs should be able to divide evenly into hp_policy_epochs

# arguments for search
perturbation_interval = True
