#!/usr/bin/env bash
# Author: cuiyiwork@foxmail.com
# Created Time: 26 March 2019 16:30

# export CUDA_VISIBLE_DEVICES="2,3"
# export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="-1"

dirr=`dirname $0`
cd ${dirr}/..

# template:
# seq2vec
# bash transformer2/run.sh train --logdir log/transformer/0 --run_type new --task_type seq2vec --train_data data/transformer/learn_bert.8k --eval_data data/transformer/learn_bert.2k --inputs 0 --vocabs vocab/vocab.3 --batch_size 128 --lr 0.006 --warmup_steps 100 --maxlens 100 --d_imitate 512 --loss_func imitate
# seq2vecseq
# bash transformer2/run.sh train --logdir log/transformer/1 --run_type new --task_type seq2vecseq --train_data data/transformer/bertout2instance.8k --eval_data data/transformer/bertout2instance.2k --inputs 0 --vocabs vocab/vocab.3 --batch_size 128 --lr 0.006 --warmup_steps 100 --maxlens 100 --d_imitate 768 --loss_func imitate_seq
if [[ "$1" = "train" ]]; then
    shift
    args="$@"
    python transformer2/train.py --d_model 128  --num_blocks 4 --d_ff 1024 ${args}
elif [[ "$1" = "infer" ]]; then
    exit 1
elif [[ "$1" = "debug" ]]; then
    shift
    args="$@"
else
    echo "Usage: bash debug.sh train/infer/debug"
fi
