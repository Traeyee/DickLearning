#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES="2,3"
# export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="-1"

dirr=`dirname $0`
cd ${dirr}/..

# template:
# bash fm/run.sh train --logdir log/fm/0 --run_type new --task_type set2sca --train_data data/fm/entity_pair.20k.inst --eval_data entity_pair.5k --inputs 0,1 --vocabs vocab/reserved_entities.txt:vocab/reserved_entities.txt --batch_size 128 --lr 0.006 --warmup_steps 100
if [[ "$1" = "train" ]]; then
    shift
    args="$@"
    python fm/train.py --d_model 128 ${args}
elif [[ "$1" = "infer" ]]; then
    exit 1
elif [[ "$1" = "debug" ]]; then
    shift
    args="$@"
else
    echo "Usage: bash debug.sh train/infer/debug"
fi
