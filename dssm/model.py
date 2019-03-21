#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 6/3/2019 2:59 PM
import tensorflow as tf
from data_load import load_vocab
from modules import noam_scheme, get_token_embeddings, ff, get_subword_embedding
from utils import get_available_gpus, average_gradients, get_gradients_by_loss_and_optimizer


class DSSM:
    def __init__(self, context):
        self.context = context
        self.token2idx, self.idx2token = load_vocab(context.vocab)
        vocab_size = len(self.token2idx)
        # 其实这里的d_model可以是其它维度
        self.embeddings = get_token_embeddings(vocab_size, self.context.d_ff, zero_pad=False)

    def unilateral_net(self, x, name, training=True):
        """
        :param x: (N, num_entities)
        :param name:
        :param training:
        :return:
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # [batch_size, seq_length, embedding_size], [vocab_size, embedding_size]
            embedding = get_subword_embedding(self.embeddings, x)  # (N, num_entities, ff_model)
            embedding = tf.reduce_mean(embedding, axis=1, name="embedding")  # (N, ff_model)

            embedding = tf.layers.dropout(embedding, self.context.dropout_rate, training=training)
            embedding = ff(embedding, [self.context.d_ff, self.context.d_ff])
            final_embedding = tf.sigmoid(tf.layers.dense(embedding, self.context.d_model))
            final_embedding = tf.identity(final_embedding, name=name+"_embedding")  # (N, num_entities, d_model)
        return final_embedding

    def _get_prediction(self, xs1, xs2):
        vec1 = self.unilateral_net(xs1, "query", True)
        vec2 = self.unilateral_net(xs2, "answer", True)
        vec1_l2 = tf.reduce_sum(tf.square(vec1), axis=1)  # (N,)
        vec2_l2 = tf.reduce_sum(tf.square(vec2), axis=1)
        predictions = tf.reduce_sum(vec1 * vec2, axis=1) / (tf.sqrt(vec1_l2 * vec2_l2))
        predictions = tf.identity(predictions, name="predictions")
        return predictions

    def train(self, xs1, xs2, scores):
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.context.lr, global_step, self.context.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        gpus = get_available_gpus()

        if gpus:
            num_gpu = len(gpus)
            assert self.context.hparams.batch_size % num_gpu == 0

            xs1s, xs2s = tf.split(xs1, num_gpu, axis=0), tf.split(xs2, num_gpu, axis=0)
            scoress = tf.split(scores, num_gpu, axis=0)

            tower_grads = []
            losses = []
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for i in range(num_gpu):
                    with tf.device("/gpu:%d" % i):
                        with tf.name_scope("tower_%d" % i):
                            predictions = self._get_prediction(xs1s[i], xs2s[i])
                            # square loss
                            partial_loss = tf.reduce_sum(tf.squared_difference(predictions, scoress[i]), name="loss")
                            losses.append(partial_loss)
                            tf.get_variable_scope().reuse_variables()
                            grad = get_gradients_by_loss_and_optimizer(partial_loss, optimizer)
                            tower_grads.append(grad)
            loss = tf.reduce_mean(losses)
            grads_and_vars = average_gradients(tower_grads)
        else:
            predictions = self._get_prediction(xs1, xs2)
            loss = tf.reduce_sum(tf.squared_difference(predictions, scores), name="loss")
            grads_and_vars = get_gradients_by_loss_and_optimizer(loss, optimizer)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        for g, v in grads_and_vars:
            tf.summary.histogram(v.name, v)
            tf.summary.histogram(v.name + '_grad', g)
        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()
        return loss, train_op, global_step, summaries
