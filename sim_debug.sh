#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="2,3"
# export CUDA_VISIBLE_DEVICES="0,1"
# export CUDA_VISIBLE_DEVICES="-1"

if [ "$1" = "train" ]; then
    shift 
    args="$@"
    python sim_train.py --d_model 128 --num_blocks 3 --hidden_size 128 $args
elif [ "$1" = "infr" ]; then
    exit 1
elif [ "$1" = "debug" ]; then
    shift 
    args="$@"
    python sim_train.py $args
else
    echo "Usage: bash debug.sh train/infr/debug"
fi

