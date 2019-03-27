#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 6/3/2019 2:59 PM
import tensorflow as tf
from base_model import BaseModel
from modules import get_subword_embedding
from utils import convert_idx_to_token_tensor


class FM(BaseModel):
    def __init__(self, context, name="fm"):
        super(FM, self).__init__(context, name=name)
        self.set_activation(tf.sigmoid)
        self._embedding_dim = context.d_model

        self._init_embeddings()

    def _infer(self, inputs, training):
        assert len(inputs) == len(self._embeddings)

        vecs_emb = [get_subword_embedding(self._embeddings[_i], _input) for _i, _input in enumerate(inputs)]
        vecs = [tf.reduce_sum(_vec, axis=1) for _vec in vecs_emb]
        sum_vec = tf.add_n(vecs)
        inferences = tf.reduce_sum(tf.square(sum_vec), axis=1)
        for vec in vecs:
            inferences -= tf.reduce_sum(tf.square(vec), axis=1)
        inferences /= 2.0
        return inferences

    def _get_loss(self, inputs, targets):
        inferences = self.infer(inputs, training=True)
        activated_infr = self.activate(inferences)
        loss = tf.reduce_mean(tf.squared_difference(activated_infr, targets), name="loss")
        return loss

    def eval(self, inputs, targets):
        inferences = self.infer(inputs)
        activated_infr = self.activate(inferences)

        n = tf.random_uniform((), 0, tf.shape(inferences)[0] - 1, tf.int32)
        tfstrings = [tf.convert_to_tensor("pred:", tf.string) + tf.as_string(activated_infr[n])]
        for i, inputt in enumerate(inputs):
            tokens = convert_idx_to_token_tensor(inputt[n], self._idx2tokens[i])
            prefix = tf.convert_to_tensor("input_%s:" % i, tf.string)
            tfstrings.append(prefix + tokens)
        tf.summary.text("eval result", tf.strings.join(tfstrings, separator="|||"))
        summaries = tf.summary.merge_all()
        return activated_infr, summaries
