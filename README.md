## A Refactored Version Of Population Based Augmentation  
## 特征  
支持自定义模型  
支持自定义数据集  
支持自定义搜索空间  
增加val_loss指标  
支持多线程载入和处理数据，解决cpu单线程执行Population Based Augmentation的性能瓶颈  
无需将所有数据载入内存，解决数据集较大时程序内存不足的问题  
## 安装和配置环境  
pip install -r requirements.txt  
## 使用方法  
### 自定义模型  
配置models/model_config.py中的build_model  
格式：  
```
def build_efficientnet(inputs, num_classes, is_training)
"""

Args:
 inputs: tf.Tensor
 num_classes: int
 is_training: bool

Returns:
 logits 
"""
```
#### 为什么不将模型作为超参数统一到配置文件中？  
PBT需要存储超参数为JSON格式，而function类型是不可JSON序列化的。  
### 自定义搜索空间  
#### 通过配置HP_TRANSFORM_NAMES定义搜索空间  
从auto_contrast, equalize, invert, rotate, posterize, crop_bilinear, solarize, color, contrast, brightness, sharpness, shear_x, shear_y, translate_x, translate_y, cutout, blur, smooth中选择想要的增强
#### 配置数据增强的映射范围  
```
rotate_max_degree = 30 [-max,max]
posterize_max = 4 [0, max]
enhance_max = 1.8 [0.1, max+0.1]
shear_x_max = 0.3 [-max, max]
shear_y_max = 0.3 [-max, max]
translate_x_max = 10 [-max, max]
translate_y_max = 10 [-max, max]
cutout_max_size = 20 [0,max]
```
### 自定义数据集  
```
# dataset
dataset_type = 'custom' # choose from 'svhn', 'cifar' or 'custom'
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
num_workers = 5 # number of cpu used for loading data
image_size = 224
num_classes = 2
```
### 执行搜索或训练  
```
python -m pba.train configs/efficientnet_train_jdb_sensetime.py
python -m pba.search configs/efficientnet_train_jdb_sensetime.py
```
搜索的epoch数可以小于训练的epoch数，但应可以被其整除。  



