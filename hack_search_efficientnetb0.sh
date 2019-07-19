#!/bin/bash
export PYTHONPATH="$(pwd)"
nohup \
python pba/search.py \
    --local_dir "/data/search_results/" \
    --model_name efficientnet-b0 \
    --num_samples 16 --perturbation_interval 3 --epochs 100 \
    --name efficientnetb0_search\
    --lr 0.1 --wd 0.0005 --bs 128 --test_bs 128\
    --cpu 5\
    --gpu 1\
 >> efficientnetb0_search.txt 2>&1 &\
