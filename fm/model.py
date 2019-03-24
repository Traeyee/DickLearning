#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 6/3/2019 2:59 PM
import tensorflow as tf
from data_load import load_vocab
from base_model import BaseModel
from modules import noam_scheme, get_token_embeddings, get_subword_embedding
from utils import get_available_gpus, average_gradients, get_gradients_by_loss_and_optimizer, \
    convert_idx_to_token_tensor


class FM2(BaseModel):
    def __init__(self, context):
        super(FM2, self).__init__(context)
        self.set_activation(tf.sigmoid)

        self.token2idxs, self.idx2tokens = [], []
        self.embeddings = []
        for i, vocab in enumerate(context.vocabs.split(":")):
            token2idx, idx2token = load_vocab(vocab)
            self.token2idxs.append(token2idx)
            self.idx2tokens.append(idx2token)
            vocab_size = len(token2idx)
            embedding = get_token_embeddings(vocab_size, context.d_model, zero_pad=False,
                                                         name="fm_{}".format(context.embedding_name[i]))
            self.embeddings.append(embedding)

    def _infer(self, inputs):
        assert len(inputs) == len(self.embeddings)

        fm_inputs = [tf.identity(_input, "fm_input_%s" % _i) for _i, _input in enumerate(inputs)]
        vecs_emb = [get_subword_embedding(self.embeddings[_i], _input) for _i, _input in enumerate(fm_inputs)]
        vecs = [tf.reduce_sum(_vec, axis=1) for _vec in vecs_emb]
        sum_vec = tf.add_n(vecs)
        inferences = tf.reduce_sum(tf.square(sum_vec), axis=1)
        for vec in vecs:
            inferences -= tf.reduce_sum(tf.square(vec), axis=1)
        inferences /= 2.0

        self._inferences = inferences
        return inferences

    def _get_loss(self, inputs, targets):
        inferences = self.infer(inputs)
        activated_infr = self.activate(inferences)
        self._outputs = activated_infr
        loss = tf.reduce_mean(tf.squared_difference(activated_infr, targets), name="loss")
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
        input1 = tf.identity(xs1, "query_indices")
        input2 = tf.identity(xs2, "answer_indices")

        query_embedding = get_subword_embedding(self.query_embeddings, input1)  # (N, num_entities, d_model)
        query_embedding = tf.reduce_sum(query_embedding, axis=1, name="query_embedding")

        answer_embedding = get_subword_embedding(self.answer_embeddings, input2)  # (N, num_entities, d_model)
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
            loss = tf.reduce_mean(losses, name="loss")
            grads_and_vars = average_gradients(tower_grads)
        else:
            predictions = self._get_prediction(xs1, xs2)
            loss = tf.reduce_mean(tf.squared_difference(predictions, scores), name="loss")
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
