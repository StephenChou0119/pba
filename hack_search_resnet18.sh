#!/bin/bash
export PYTHONPATH="$(pwd)"

python pba/search.py \
    --local_dir "$PWD/results/" \
    --model_name resnet \
    --num_samples 8 --perturbation_interval 3 --epochs 100 \
    --name resnet_100\
    --lr 0.1 --wd 0.0005 --bs 32 --test_bs 32\
    --cpu 5\
    --gpu 1\
    --resnet_size 20\
