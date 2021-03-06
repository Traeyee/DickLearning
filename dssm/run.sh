#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES="2,3"
export CUDA_VISIBLE_DEVICES="0,1"
# export CUDA_VISIBLE_DEVICES="-1"

dirr=`dirname $0`
cd ${dirr}/..

if [[ "$1" = "train" ]]; then
    shift
    args="$@"
    python3 dssm/train.py --d_model 128 --d_ff 256 --vocab vocab/reserved_entities.txt ${args}
elif [[ "$1" = "infr" ]]; then
    exit 1
elif [[ "$1" = "debug" ]]; then
    shift
    args="$@"
    python dssm_train.py ${args}
else
    echo "Usage: bash debug.sh train/infr/debug"
fi

