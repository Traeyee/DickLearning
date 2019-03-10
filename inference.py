#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 4/3/2019 11:32 AM
import os
import logging
import time
import tensorflow as tf
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

from context import Context
from transformer.hparams import Hparams
from transformer.model import Transformer
from transformer.utils import get_hypotheses
from transformer.data_load import get_batch

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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

context = Context(hp)
m = Transformer(context)
y_hat, eval_summaries = m.eval(xs, ys)  # y_hat: elements are indices, a target that can be run


logging.info("# Session")
saver = tf.train.Saver()
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.warning("No checkpoint is found")
        exit(1)
    else:
        saver.restore(sess, ckpt)

    logging.info("# test evaluation")
    sess.run(eval_init_op)
    # It means 跑op=y_hat，输入是num_eval_batches，只截取前num_eval_samples个结果
    logging.info("# get hypotheses")
    if hp.use_profile:
        logging.info("# init profile")
        run_metadata = tf.RunMetadata()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        mnist_profiler = model_analyzer.Profiler(graph=sess.graph)
        ts = time.time()
        hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.idx2token,
                                    use_profile=True,
                                    options=run_options, run_metadata=run_metadata, profiler=mnist_profiler)
        logging.info("eval: takes %s" % (time.time() - ts))
        # 统计内容为每个graph node的运行时间和占用内存
        profile_graph_opts_builder = option_builder.ProfileOptionBuilder(
            option_builder.ProfileOptionBuilder.time_and_memory())

        # 输出方式为timeline
        profile_graph_opts_builder.with_timeline_output(timeline_file='/tmp/mnist_profiler.json')
        # 定义显示sess.Run() 第0步的统计数据
        profile_graph_opts_builder.with_step(0)
        profile_graph_opts_builder.with_step(1)
        # 显示视图为graph view
        mnist_profiler.profile_graph(profile_graph_opts_builder.build())
    else:
        ts = time.time()
        hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.idx2token)
        logging.info("eval: takes %s" % (time.time() - ts))
    if not os.path.exists(hp.evaldir):
        os.makedirs(hp.evaldir)
    translation = os.path.join(hp.evaldir, "inference.out")
    with open(translation, 'w') as fout:
        fout.write("\n".join(hypotheses))
