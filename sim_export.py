#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 11/3/2019 2:37 PM
import logging
import os
import tensorflow as tf

from context import Context
from transformer.sim_hparams import SimHparams as Hparams
from transformer.sim_model import SimTransformer
from transformer.utils import load_hparams, save_operation_specs

logger = logging.getLogger()
logger.setLevel(logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp, hp.logdir)
context = Context(hp)

m = SimTransformer(context)
xs = (tf.placeholder(dtype=tf.int32, shape=[None, None], name="xs"), tf.constant(0), tf.constant(0))
vec = m.get_encodec(xs)

saver = tf.train.Saver()
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    saver.restore(sess, ckpt)
    graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=["seq2vec/vec"])
    tf.train.write_graph(graph_def, './model', '%s.pb' % hp.pb_name, as_text=False)
    save_operation_specs(os.path.join("./model", '%s.ops' % hp.pb_name))
