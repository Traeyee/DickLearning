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
        else:
            assert hparams.vocab is not None or hparams.vocabs, "Neither use_auto_vocab and vocab cannot be negative"

        # version2
        self.vocabs = hparams.vocabs
        self.input_indices = hparams.inputs
        self.embedding_name = None
        if self.vocabs or self.input_indices:
            assert self.vocabs.count(":") == self.input_indices.count(",")
            self.embedding_name = ["input_embedding_%s" % _i for _i in self.input_indices.split(",")]
        self.embed_token_idx = (None, None, None)  # 用已有的embedding初始化模型
        self.embedding_dims = None  # 指定每个embedding的dim
        self.embedded_indices = None  # 指定只对哪些vocab进行embedding
        self.maxlens = []
        if hparams.maxlens:
            for maxlen in hparams.maxlens.split(","):
                self.maxlens.append(int(maxlen))
        self.loss_func = hparams.loss_func
        self.d_imitate = hparams.d_imitate

    def set_vocabs(self, vocabs):
        self.vocabs = vocabs
