#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES="2,3"
# export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="-1"

dirr=`dirname $0`
cd ${dirr}/..

# template:
# bash lr/run.sh train --logdir log/lr/1 --run_type new --train_data data/fm/entity_pair.20k.inst --inputs 0 --vocabs vocab/reserved_entities.txt --batch_size 1024 --lr 0.006 --warmup_steps 100
if [[ "$1" = "train" ]]; then
    shift
    args="$@"
    python lr/train.py ${args}
elif [[ "$1" = "infer" ]]; then
    exit 1
elif [[ "$1" = "debug" ]]; then
    shift
    args="$@"
else
    echo "Usage: bash debug.sh train/infer/debug"
fi
