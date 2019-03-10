#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 6/3/2019 2:59 PM
import tensorflow as tf
from transformer.model import Transformer
from transformer.modules import noam_scheme
from transformer.utils import get_available_gpus, average_gradients


class SimTransformer(Transformer):
    def __init__(self, context):
        super(SimTransformer, self).__init__(context)

    def seq2vec(self, inputs, name, training=False):
        """
        :param inputs: (N, T, d_model)
        :param name: output tensor name
        :param training: if in training phase
        :return: (N, d_model)
        """
        with tf.variable_scope("seq2vec", reuse=tf.AUTO_REUSE):
            attention_vector = tf.get_variable("attention_matrix", shape=[self.context.d_model])
            attention_weight = tf.einsum("nij,j->ni", inputs, attention_vector)  # (N, T)
            attention_weight = tf.nn.softmax(attention_weight, name="attention_weight")  # (N, T)

            # Mask
            masks = tf.sign(tf.reduce_sum(tf.abs(inputs), axis=-1))  # (N, T)
            attention_weight *= masks  # broadcasting. (N, T)
            attention_weight = tf.identity(attention_weight, name="attention_weight_masked")

            # Dropouts
            attention_weight = tf.layers.dropout(attention_weight, rate=self.context.dropout_rate,
                                                 training=tf.convert_to_tensor(training))

            outputs = tf.einsum("nij,ni->nj", inputs, attention_weight)
            outputs = tf.identity(outputs, name=name)
        return outputs

    def sim_train(self, xs1, xs2, scores):
        memory1, sents1 = self.encode(xs1, name="xs1")  # (N, T1, d_model)
        memory2, sents2 = self.encode(xs2, name="xs2")
        vec1 = self.seq2vec(memory1, "vec1", True)  # (N, d_model)
        vec2 = self.seq2vec(memory2, "vec2", True)
        vec1_l2 = tf.reduce_sum(tf.square(vec1), axis=1)  # (N,)
        vec2_l2 = tf.reduce_sum(tf.square(vec2), axis=1)
        predictions = tf.reduce_sum(vec1 * vec2, axis=1) / (tf.sqrt(vec1_l2 * vec2_l2))
        predictions = tf.identity(predictions, name="predictions")

        loss = tf.reduce_sum(tf.squared_difference(predictions, scores), name="loss")
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.context.lr, global_step, self.context.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)

        # train_op = optimizer.minimize(loss, global_step=global_step)
        # 对optimizer.minimize进行展开
        grads_and_vars = optimizer.compute_gradients(loss)
        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in grads_and_vars], loss))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        for g, v in grads_and_vars:
            tf.summary.histogram(v.name, v)
            tf.summary.histogram(v.name + '_grad', g)
        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()
        return loss, train_op, global_step, summaries

    def sim_train_multigpu(self, xs1, xs2, scores):
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.context.lr, global_step, self.context.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        gpus = get_available_gpus()

        if gpus:
            num_gpu = len(gpus)
            assert self.context.hparams.batch_size % num_gpu == 0

            def split_data(xs, num):
                x, seqlens, sents = xs
                xss = []
                _x = tf.split(x, num, axis=0)
                _seqlens = tf.split(seqlens, num, axis=0)
                _sents = tf.split(sents, num, axis=0)
                for k in range(num):
                    xss.append((_x[k], _seqlens[k], _sents[k]))
                return xss
            xs1s, xs2s = split_data(xs1, num_gpu), split_data(xs2, num_gpu)
            scoress = tf.split(scores, num_gpu, axis=0)

            tower_grads = []
            losses = []
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for i in range(num_gpu):
                    with tf.device("/gpu:%d" % i):
                        with tf.name_scope("tower_%d" % i):
                            grad, single_loss = self._get_gradients(xs1s[i], xs2s[i], scoress[i], optimizer)
                            losses.append(single_loss)
                            tower_grads.append(grad)
            loss = tf.reduce_mean(losses)
            grads_and_vars = average_gradients(tower_grads)
        else:
            grads_and_vars, loss = self._get_gradients(xs1, xs2, scores, optimizer)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        for g, v in grads_and_vars:
            tf.summary.histogram(v.name, v)
            tf.summary.histogram(v.name + '_grad', g)
        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()
        return loss, train_op, global_step, summaries

    def _get_gradients(self, xs1, xs2, scores, optimizer):
        memory1, sents1 = self.encode(xs1, name="xs1")  # (N, T1, d_model)
        memory2, sents2 = self.encode(xs2, name="xs2")
        vec1 = self.seq2vec(memory1, "vec1", True)  # (N, d_model)
        vec2 = self.seq2vec(memory2, "vec2", True)
        vec1_l2 = tf.reduce_sum(tf.square(vec1), axis=1)  # (N,)
        vec2_l2 = tf.reduce_sum(tf.square(vec2), axis=1)
        predictions = tf.reduce_sum(vec1 * vec2, axis=1) / (tf.sqrt(vec1_l2 * vec2_l2))
        predictions = tf.identity(predictions, name="predictions")

        loss = tf.reduce_sum(tf.squared_difference(predictions, scores), name="loss")
        tf.get_variable_scope().reuse_variables()
        grads_and_vars = optimizer.compute_gradients(loss)
        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in grads_and_vars], loss))
        return grads_and_vars, loss
