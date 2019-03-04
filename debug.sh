#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="2,3"

if [ "$1" = "train" ]; then
    python train.py --num_epochs 22
elif [ "$1" = "infr" ]; then
    python inference.py --num_epochs 22
elif [ "$1" = "debug" ]; then
    python debug.py --num_epochs 22 --eval_batch_size 4
else
    echo "Usage: bash debug.sh train/infr/debug"
fi

