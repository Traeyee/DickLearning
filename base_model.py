#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 24 March 2019 14:46
import tensorflow as tf
from abc import abstractmethod

from modules import noam_scheme
from utils import get_available_gpus, average_gradients, get_gradients_by_loss_and_optimizer


class BaseModel:
    def __init__(self, context, name="model"):
        self._context = context
        self._name = name
        self.embeddings = None

        self._inferences = None
        self._outputs = None
        self._activation = None

    @abstractmethod
    def _infer(self, inputs):
        pass

    @abstractmethod
    def eval(self, inputs, targets):
        """每个模型的eval目标不一样"""
        pass

    @abstractmethod
    def _get_loss(self, inputs, targets):
        return 0.0

    def infer(self, inputs):
        inferences = self._infer(inputs)
        inferences = tf.identity(inferences, "inferences")
        self._inferences = inferences
        self._outputs = self.activate(inferences)
        return inferences

    def train(self, inputs, targets):
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self._context.lr, global_step, self._context.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        gpus = get_available_gpus()

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
                            partial_loss = self._get_loss(partial_inputs[i], targetses[i])
                            losses.append(partial_loss)
                            tf.get_variable_scope().reuse_variables()
                            grad = get_gradients_by_loss_and_optimizer(partial_loss, optimizer)
                            tower_grads.append(grad)
            loss = tf.reduce_mean(losses)
            grads_and_vars = average_gradients(tower_grads)
        else:
            loss = tf.reduce_mean(self._get_loss(inputs, targets))
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
        self.embeddings = embeddings
