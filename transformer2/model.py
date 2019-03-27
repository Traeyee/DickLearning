#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 26 March 2019 14:54
import logging
import tensorflow as tf
from base_model import BaseModel
from modules import positional_encoding, ff, multihead_attention


class Transformer(BaseModel):
    def __init__(self, context, name="transformer"):
        super(Transformer, self).__init__(context, name=name)
        self.set_activation(lambda x: x)
        self._embedding_dim = context.d_model

        self._init_embeddings()
        self._loss_func_dict = {"imitate": self._get_loss_imitate, "imitate_seq": self._get_loss_imitate_seq}


    def _infer(self, inputs, training):
        """
        Returns
        memory: encoder outputs. (N, T1, d_model)
        """
        assert len(inputs) == len(self._embeddings)
        assert len(inputs) == len(self._context.maxlens)
        logging.info("num_blocks:%s", self._context.num_blocks)
        # 目前只有一个输出的场景
        memories = self._encode(inputs[0], 0, training=training, name="encode_input_0")
        return memories

    def _infer_seq(self, inputs, training):
        assert False, "Not implemented yet"
        memories = self._infer(inputs, training)

    def _infer_vec(self, inputs, training):
        memories = self._infer(inputs, training)
        first_token_tensor = tf.squeeze(memories[:, 0:1, :], axis=1)
        return first_token_tensor

    def _get_loss(self, inputs, targets):
        """为_infer_seq而预留"""
        assert False, "Not implemented yet"
        return 0.0

    def _get_loss_imitate(self, inputs, targets):
        inferences = self.infer(inputs, training=True, infer=self._infer_vec)
        outputs = tf.layers.dense(inputs=inferences, units=self._context.d_imitate, reuse=tf.AUTO_REUSE)  # (N, d_imitate)
        loss = tf.reduce_mean(tf.squared_difference(outputs, targets), name="loss")
        return loss

    def _get_loss_imitate_seq(self, inputs, targets):
        inferences = self.infer(inputs, training=True, infer=self._infer)  # (N, T1, d_model)
        outputs = tf.layers.dense(inputs=inferences, units=self._context.d_imitate, reuse=tf.AUTO_REUSE)  # (N, T1, d_imitate)
        loss = tf.reduce_mean(tf.squared_difference(outputs, targets), name="loss")
        return loss

    def eval(self, inputs, targets):
        assert self._context.loss_func in ("imitate", "imitate_seq"), "Other infer is not implemented yet"
        if "imitate" == self._context.loss_func:
            inferences = self.infer(inputs, training=False, infer=self._infer_vec)
        elif "imitate_seq" == self._context.loss_func:
            inferences = self.infer(inputs, training=False, infer=self._infer)
        summaries = tf.summary.merge_all()
        return 0.0, summaries

    def _encode(self, x, seq_num, training=True, name=None):
        """
        Returns
        memory: encoder outputs. (N, T1, d_model)
        """
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # embedding
            x = tf.identity(x, "input_x")
            enc = tf.nn.embedding_lookup(self._embeddings[seq_num], x)   # (N, T1, d_model)
            enc *= self._context.d_model**0.5  # scale

            enc += positional_encoding(enc, self._context.maxlens[seq_num])
            enc = tf.layers.dropout(enc, self._context.dropout_rate, training=training)

            # # Blocks
            for i in range(self._context.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self._context.num_heads,
                                              dropout_rate=self._context.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self._context.d_ff, self._context.d_model])
        memory = tf.identity(enc, name=name)
        return memory
