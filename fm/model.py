#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 6/3/2019 2:59 PM
import tensorflow as tf
from data_load import load_vocab
from modules import noam_scheme, get_token_embeddings, get_subword_embedding
from utils import get_available_gpus, average_gradients, get_gradients_by_loss_and_optimizer


class FM:
    def __init__(self, context):
        self.context = context
        self.token2idx, self.idx2token = load_vocab(context.vocab)
        vocab_size = len(self.token2idx)
        # 其实这里的d_model可以是其它维度
        self.query_embeddings = get_token_embeddings(vocab_size, self.context.d_model, zero_pad=False,
                                                     name="query_entity")
        self.answer_embeddings = get_token_embeddings(vocab_size, self.context.d_model, zero_pad=False,
                                                      name="answer_entity")

    def _get_prediction(self, xs1, xs2, training=True):
        query_embedding = get_subword_embedding(self.query_embeddings, xs1)  # (N, num_entities, d_model)
        query_embedding = tf.reduce_sum(query_embedding, axis=1, name="query_embedding")

        answer_embedding = get_subword_embedding(self.answer_embeddings, xs2)  # (N, num_entities, d_model)
        answer_embedding = tf.reduce_sum(answer_embedding, axis=1, name="answer_embedding")

        total_embedding = query_embedding + answer_embedding  # (N, d_model)
        predictions = tf.reduce_sum(tf.square(total_embedding), axis=1)
        predictions -= tf.reduce_sum(tf.square(query_embedding), axis=1)
        predictions -= tf.reduce_sum(tf.square(answer_embedding), axis=1)
        predictions /= 2.0
        predictions = tf.sigmoid(predictions)

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
                list_predictions = []
                for i in range(num_gpu):
                    with tf.device("/gpu:%d" % i):
                        with tf.name_scope("tower_%d" % i):
                            predictions = self._get_prediction(xs1s[i], xs2s[i])
                            list_predictions.append(predictions)
                            # square loss
                            partial_loss = tf.reduce_sum(tf.squared_difference(predictions, scoress[i]), name="loss")
                            losses.append(partial_loss)
                            tf.get_variable_scope().reuse_variables()
                            grad = get_gradients_by_loss_and_optimizer(partial_loss, optimizer)
                            tower_grads.append(grad)
                predictions = tf.concat(list_predictions, axis=0)
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
        tf.summary.scalar("pred_avg", tf.reduce_mean(predictions))
        tf.summary.scalar("label_avg", tf.reduce_mean(scores))

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()
        return loss, train_op, global_step, summaries
