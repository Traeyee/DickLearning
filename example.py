#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 24 March 2019 21:07
import tensorflow as tf

from base_model import BaseModel
from fm.model import FM2
from lr.model import LR
from schemes import train_template


class EgModel(BaseModel):
    def __init__(self, context):
        super(EgModel, self).__init__(context)
        self.fm = FM2(context)
        lr = LR(context)
        lr.set_embeddings(lr.embeddings[1:])
        self.lr = lr
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
        self._outputs = activated_infr
        loss = tf.reduce_mean(tf.squared_difference(activated_infr, targets), name="loss")
        return loss

train_template(EgModel)
