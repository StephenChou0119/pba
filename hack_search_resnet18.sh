#!/bin/bash
export PYTHONPATH="$(pwd)"

python pba/search.py \
    --local_dir "$PWD/results/" \
    --model_name resnet \
    --num_samples 16 --perturbation_interval 3 --epochs 100 \
    --name resnet_100\
    --lr 0.1 --wd 0.0005 --bs 48 --test_bs 48\
    --cpu 5\
    --gpu 1\
    --resnet_size 20\
