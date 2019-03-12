#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 6/3/2019 3:04 PM
import logging
import os
import time
import math
import tensorflow as tf
from tqdm import tqdm

from context import Context
from transformer.data_load import get_batch_sim
from transformer.sim_hparams import SimHparams as Hparams
from transformer.sim_model import SimTransformer
from transformer.utils import save_hparams, save_variable_specs, save_operation_specs, load_hparams

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
context = Context(hp)

logging.info("# Prepare train/eval batches")
logging.info("Use %s for training set", hp.train_sim)
train_batches, num_train_batches, num_train_samples = get_batch_sim(hp.train_sim, hp.maxlen1, hp.maxlen2,
                                                                    context.vocab, hp.batch_size, shuffle=True)

# create a iterator of the correct shape and type
iterr = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
x1, x2, score = iterr.get_next()

# 照抄即可，目前不是很熟悉这些接口
train_init_op = iterr.make_initializer(train_batches)

m = SimTransformer(context)
loss, train_op, global_step, train_summaries = m.sim_train(x1, x2, score)

f_debug = open(os.path.join(hp.logdir, "debug.txt"), "a")
logging.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    time_sess = time.time()
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        save_hparams(hp, hp.logdir)
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, ckpt)
        if hp.fine_tune:
            save_hparams(hp, hp.logdir)
        else:
            load_hparams(hp, hp.logdir)
    save_variable_specs(os.path.join(hp.logdir, "var_specs"))
    save_operation_specs(os.path.join(hp.logdir, "op_specs"))
    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)
    if hp.zero_step:
        sess.run(global_step.assign(0))

    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)
    logging.info("global_step is stated at %s", _gs)
    t_epoch = time.time()
    model_output = 'default'
    for i in tqdm(range(_gs, total_steps + 1)):
        ts = time.time()
        # f_debug.write("loss\n")
        # tensor_tmp = tf.get_default_graph().get_tensor_by_name("loss:0")
        # np.savetxt(f_debug, tensor_tmp.eval().reshape([1]), delimiter=', ', footer="=" * 64)
        _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
        epoch = math.ceil(_gs / num_train_batches)
        f_debug.write("train: epoch %s takes %s" % (epoch, time.time() - ts))
        summary_writer.add_summary(_summary, _gs)

        if _gs and _gs % num_train_batches == 0:
            logging.info("epoch {} is done".format(epoch))
            _loss = sess.run(loss)  # train loss

            logging.info("# save models")
            model_output = "sim%02dL%.2f" % (epoch, _loss)
            ckpt_name = os.path.join(hp.logdir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# fall back to train mode")
            ts = time.time()
            sess.run(train_init_op)
            logging.info("fallback_train: %s\t%s\t%s takes %s" % (i, _gs, epoch, time.time() - ts))
            logging.info("epoch %s takes %s", epoch, time.time() - t_epoch)
            t_epoch = time.time()
    summary_writer.close()
    logging.info("Session runs for %s", time.time() - time_sess)
    graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=["seq2vec/vec1"])
    tf.train.write_graph(graph_def, './model', '%s.pb' % model_output, as_text=False)
f_debug.close()
logging.info("Done")
