#!/bin/bash
export PYTHONPATH="$(pwd)"

python pba/search.py \
    --local_dir "$PWD/results/" \
    --model_name wrn_40_2 \
    --num_samples 8 --perturbation_interval 3 --epochs 10 \
    --name hack_first\
    --lr 0.1 --wd 0.0005 --bs 48 --test_bs 48\
    --cpu 5\
    --gpu 1\

