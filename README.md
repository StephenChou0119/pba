# Population Based Augmentation
# Goal
The project is created mainly for convenient practical application rather than research.
## Features
1. refactored Data loading module for a batch, rather than load all data into the memory.
2. support custom model
3. support custom dataset using a csv 
 
## TODO
1. currently, one model can only use one GPU, model parallel should be supported.
2. the code can only run at a server, distributed training should be supported.
3. the last batch has been dropped, it should be fixed.

## Usage  
```
nohup python pba/search.py configs/config_demo.py &
```