#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 4/3/2019 11:32 AM
import logging
import time
import tensorflow as tf
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

from transformer.hparams import Hparams
from transformer.model import Transformer
from transformer.utils import get_hypotheses
from transformer.data_load import get_batch
logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

"""(object, scalar, scalar)
注意这个maxlen“100”，太大了的话图会变得非常大
"""
eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval1, hp.eval2, hp.maxlen1, hp.maxlen2,
                                                             hp.vocab, hp.eval_batch_size, shuffle=False)
"""(x, x_seqlen, sent1), (decoder_input, y, y_seqlen, sent2)
eval_batches.output_types = ((tf.int32, tf.int32, tf.string), (tf.int32, tf.int32, tf.int32, tf.string))
eval_batches.output_shapes = (([None], (), ()), ([None], [None], (), ()))
"""
iterr = tf.data.Iterator.from_structure(eval_batches.output_types, eval_batches.output_shapes)
xs, ys = iterr.get_next()

eval_init_op = iterr.make_initializer(eval_batches)  # Create an op, but not run yet

m = Transformer(hp)
y_hat, _ = m.debug(xs, ys)  # y_hat: elements are indices, a target that can be run


logging.info("# Session")
saver = tf.train.Saver()
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.warning("No checkpoint is found")
        exit(1)
    else:
        saver.restore(sess, ckpt)
    # logging.info("# init profile")
    # run_metadata = tf.RunMetadata()
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # mnist_profiler = model_analyzer.Profiler(graph=sess.graph)

    logging.info("# test evaluation")
    sess.run(eval_init_op)
    # It means 跑op=y_hat，输入是num_eval_batches，只截取前num_eval_samples个结果
    logging.info("# get hypotheses")
    ts_total = time.time()
    for ith_batch in range(num_eval_batches):
        logging.info("db11")
        ts = time.time()
        h = sess.run(y_hat)
        logging.info("%sth takes %s", ith_batch, time.time() - ts)
    logging.info("total takes %s", time.time() - ts_total)
