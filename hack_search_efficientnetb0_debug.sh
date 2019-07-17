#!/bin/bash
export PYTHONPATH="$(pwd)"

python pba/search.py \
    --local_dir "$PWD/results/" \
    --model_name efficientnet-b0 \
    --num_samples 8 --perturbation_interval 3 --epochs 10 \
    --name resnet_100\
    --lr 0.1 --wd 0.0005 --bs 16 --test_bs 16\
    --cpu 5\
    --gpu 1\
