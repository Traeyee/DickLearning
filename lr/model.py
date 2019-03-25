#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 24 March 2019 19:38
import tensorflow as tf
from data_load import load_vocab
from base_model import BaseModel
from modules import get_token_embeddings, get_subword_embedding
from utils import convert_idx_to_token_tensor


class LR(BaseModel):
    def __init__(self, context):
        super(LR, self).__init__(context)
        self.set_activation(tf.sigmoid)

        self.token2idxs, self.idx2tokens = [], []
        self.embeddings = []
        for i, vocab in enumerate(context.vocabs.split(":")):
            token2idx, idx2token = load_vocab(vocab)
            self.token2idxs.append(token2idx)
            self.idx2tokens.append(idx2token)
            vocab_size = len(token2idx)
            embedding = get_token_embeddings(vocab_size, 1, zero_pad=False, name="lr_{}".format(context.embedding_name[i]))
            self.embeddings.append(embedding)

    def _infer(self, inputs):
        assert len(inputs) == len(self.embeddings)

        lr_inputs = [tf.identity(_input, "lr_input_%s" % _i) for _i, _input in enumerate(inputs)]
        # [[batch_size, set_length, 1], [batch_size, set_length, 1]...]
        weights = [get_subword_embedding(self.embeddings[_i], _input) for _i, _input in enumerate(lr_inputs)]
        logits = [tf.reduce_sum(_weight, axis=1) for _weight in weights]  # [[batch_size, 1], [batch_size, 1]...]
        inferences = tf.add_n(logits)  # [batch_size, 1]
        inferences = tf.squeeze(inferences, axis=1)
        return inferences

    def _get_loss(self, inputs, targets):
        """https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        For brevity, let x = logits, z = labels. The logistic loss is
            z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
          = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
          = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
          = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
          = (1 - z) * x + log(1 + exp(-x))
          = x - x * z + log(1 + exp(-x))
        For x < 0, to avoid overflow in exp(-x), we reformulate the above
            x - x * z + log(1 + exp(-x))
          = log(exp(x)) - x * z + log(1 + exp(-x))
          = - x * z + log(1 + exp(x))
        Hence, to ensure stability and avoid overflow, the implementation uses this equivalent formulation
            max(x, 0) - x * z + log(1 + exp(-abs(x)))
        """
        inferences = self.infer(inputs)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=inferences)
        loss = tf.reduce_mean(loss)
        return loss

    def eval(self, inputs, targets):
        inferences = self.infer(inputs)
        activated_infr = self.activate(inferences)

        from random import random
        n = int(random() * inferences.shape[0])
        for i, inputt in enumerate(inputs[n]):
            tokens = convert_idx_to_token_tensor(inputt, self.idx2tokens)
            tf.summary.text("input_%s" % i, tokens)
        tf.summary.text("pred", activated_infr)
        summaries = tf.summary.merge_all()
        return activated_infr, summaries
