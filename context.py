#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 5/3/2019 1:20 PM
import json

from nlptools import Encoder2 as Encoder


class Context:
    """Used for decoupling the hparams and model. And we can convey instance or object to other codes through this"""
    def __init__(self, hparams):
        self.hparams = hparams

        self.d_model = hparams.d_model
        self.maxlen1 = hparams.maxlen1
        self.maxlen2 = hparams.maxlen2
        self.dropout_rate = hparams.dropout_rate
        self.num_blocks = hparams.num_blocks
        self.num_heads = hparams.num_heads
        self.d_ff = hparams.d_ff
        self.warmup_steps = hparams.warmup_steps
        self.lr = hparams.lr

        self.vocab = hparams.vocab
        self.token2idx = {}
        self.idx2token = {}
        if hparams.use_auto_vocab:
            encoder = Encoder()
            self.token2idx, self.idx2token = encoder.get_index_dict()
            self.vocab = json.dumps(self.token2idx)
