#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 24 March 2019 21:07
import tensorflow as tf
from copy import deepcopy

from base_model import BaseModel
from fm.model import FM
from lr.model import LR
from templates import train_template, export_pb_template


class EgModel(BaseModel):
    def __init__(self, context):
        # context.embedding_dims = [context.d_model, 1]
        super(EgModel, self).__init__(context)
        fm_context = deepcopy(context)
        # fm_context.embed_token_idx = self.get_embed_token_idx()
        self.fm = FM(fm_context)
        # lr与正文inputs取[1:]相对应
        lr_context = deepcopy(context)
        # lr_context.embed_token_idx = tuple([_ele[1:] for _ele in self.get_embed_token_idx()])
        lr_context.embedded_indices = [1]
        self.lr = LR(lr_context)
        self.set_activation(tf.sigmoid)

    def _infer(self, inputs):
        fm_inputs = inputs
        lr_inputs = inputs[1:]
        return self.fm.infer(fm_inputs) + self.lr.infer(lr_inputs)

    def eval(self, inputs, targets):
        summaries = tf.summary.merge_all()
        infr = self.activate(self.infer(inputs))
        return infr, summaries

    def _get_loss(self, inputs, targets):
        inferences = self.infer(inputs)
        activated_infr = self.activate(inferences)
        loss = tf.reduce_mean(tf.squared_difference(activated_infr, targets), name="loss")
        return loss


def train_example():
    train_template(EgModel)


def export_pb_example():
    export_pb_template(EgModel)

if __name__ == '__main__':
    train_example()
