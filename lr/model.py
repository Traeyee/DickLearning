#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 24 March 2019 19:38
import tensorflow as tf
from base_model import BaseModel
from modules import get_subword_embedding
from utils import convert_idx_to_token_tensor


class LR(BaseModel):
    def __init__(self, context, name="lr"):
        super(LR, self).__init__(context, name=name)
        self.set_activation(tf.sigmoid)
        self._embedding_dim = 1

        self._init_embeddings()

    def _infer(self, inputs):
        assert len(inputs) == len(self._embeddings)

        # [[batch_size, set_length, 1], [batch_size, set_length, 1]...]
        weights = [get_subword_embedding(self._embeddings[_i], _input) for _i, _input in enumerate(inputs)]
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

        n = tf.random_uniform((), 0, tf.shape(inferences)[0] - 1, tf.int32)
        tfstrings = [tf.convert_to_tensor("pred:", tf.string) + tf.as_string(activated_infr[n])]
        for i, inputt in enumerate(inputs):
            tokens = convert_idx_to_token_tensor(inputt[n], self._idx2tokens[i])
            prefix = tf.convert_to_tensor("input_%s:" % i, tf.string)
            tfstrings.append(prefix + tokens)
        tf.summary.text("eval result", tf.strings.join(tfstrings, separator="|||"))
        summaries = tf.summary.merge_all()
        return activated_infr, summaries
