#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 24 March 2019 14:46
import logging
import tensorflow as tf
from abc import abstractmethod

from modules import noam_scheme, get_token_embeddings
from utils import get_available_gpus, average_gradients, get_gradients_by_loss_and_optimizer
from data_load import load_vocab


class BaseModel:
    def __init__(self, context, name="base"):
        self._context = context
        self._name = name

        self._inferences = None
        self._outputs = None
        self._activation = None

        self._embeddings = context.embed_token_idx[0]
        self._token2idxs = context.embed_token_idx[1]
        self._idx2tokens = context.embed_token_idx[2]

        self._embedding_dim = None
        self._init_embeddings()

        self._loss_func_dict = {"default": self._get_loss}

    @abstractmethod
    def eval(self, inputs, targets):
        """每个模型的eval关心的metrics不一样"""
        pass

    @abstractmethod
    def _infer(self, inputs, training):
        pass

    @abstractmethod
    def _get_loss(self, inputs, targets):
        """默认loss函数"""
        return 0.0

    def infer(self, inputs, training=False, infer=None):
        if infer is None:
            infer = self._infer
        named_inputs = [tf.identity(_input, "%s_input_%s" % (self._name, _i)) for _i, _input in enumerate(inputs)]
        inferences = infer(named_inputs, training=training)
        inferences = tf.identity(inferences, "%s_inferences" % self._name)
        self._inferences = inferences
        self._outputs = tf.identity(self.activate(inferences), "%s_outputs" % self._name)
        return inferences

    def train(self, inputs, targets):
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self._context.lr, global_step, self._context.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        gpus = get_available_gpus()

        loss_func = self._loss_func_dict.get(self._context.loss_func, self._get_loss)
        if gpus:
            num_gpu = len(gpus)
            assert self._context.hparams.batch_size % num_gpu == 0

            partial_inputs = [[] for _ in range(num_gpu)]
            for input_tmp in inputs:
                input_tmps = tf.split(input_tmp, num_gpu, axis=0)
                for i in range(num_gpu):
                    partial_inputs[i].append(input_tmps[i])
            targetses = tf.split(targets, num_gpu, axis=0)

            tower_grads = []
            losses = []
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for i in range(num_gpu):
                    with tf.device("/gpu:%d" % i):
                        with tf.name_scope("tower_%d" % i):
                            partial_loss = loss_func(partial_inputs[i], targetses[i])
                            losses.append(partial_loss)
                            tf.get_variable_scope().reuse_variables()
                            grad = get_gradients_by_loss_and_optimizer(partial_loss, optimizer)
                            tower_grads.append(grad)
            loss = tf.reduce_mean(losses)
            grads_and_vars = average_gradients(tower_grads)
        else:
            loss = tf.reduce_mean(loss_func(inputs, targets))
            grads_and_vars = get_gradients_by_loss_and_optimizer(loss, optimizer)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        for g, v in grads_and_vars:
            if g is None:  # 无梯度
                continue
            tf.summary.histogram(v.name, v)
            tf.summary.histogram(v.name + '_grad', g)
        tf.summary.scalar("pred_avg", tf.reduce_mean(self._outputs))
        tf.summary.scalar("infr_avg", tf.reduce_mean(self._inferences))
        tf.summary.scalar("label_avg", tf.reduce_mean(targets))

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()
        return loss, train_op, global_step, summaries

    def set_activation(self, activation):
        self._activation = activation

    def activate(self, inferences):
        return self._activation(inferences)

    def get_name(self):
        return self._name

    def set_embeddings(self, embeddings):
        self._embeddings = embeddings

    def get_embed_token_idx(self):
        return self._embeddings, self._token2idxs, self._idx2tokens

    def get_inference_op_name(self):
        return self._inferences.name

    def _init_embeddings(self):
        if self._embeddings is None:
            if self._embedding_dim is None and self._context.embedding_dims is None:
                logging.info("%s embedding is not initialized", self._name)
                return
            logging.info("%s embedding is being initialized", self._name)
            self._embeddings = []
            self._token2idxs = []
            self._idx2tokens = []
            cnt = 0
            for i, vocab in enumerate(self._context.vocabs.split(":")):
                if self._context.embedded_indices is not None:
                    if i not in self._context.embedded_indices:
                        continue
                token2idx, idx2token = load_vocab(vocab)
                self._token2idxs.append(token2idx)
                self._idx2tokens.append(idx2token)
                vocab_size = len(token2idx)
                dim = self._embedding_dim
                if self._context.embedding_dims is not None:
                    dim = self._context.embedding_dims[cnt]
                assert dim is not None
                embedding = get_token_embeddings(vocab_size, dim, zero_pad=False,
                                                 name="{}_{}".format(self._name, self._context.embedding_name[i]))
                self._embeddings.append(embedding)
                cnt += 1
            logging.info("%s initialized %s embeddings", self._name, cnt)

        # CHECK
        assert self._token2idxs is not None and self._idx2tokens is not None
        assert len(self._embeddings) == len(self._token2idxs) and len(self._token2idxs) == len(self._idx2tokens)
        for i in range(len(self._embeddings)):
            assert self._embeddings[i].shape[0] == len(self._token2idxs[i]), \
                "%s != %s" % (self._embeddings[i].shape[0], len(self._token2idxs[i]))
