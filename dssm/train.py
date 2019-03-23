#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 6/3/2019 3:04 PM
import logging
import os
import time
import math
import sys
import tensorflow as tf
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "./.."))
from context import Context
from hparams import Hparams
from utils import save_hparams, save_variable_specs, save_operation_specs, load_hparams
from data_load import get_batch
from dssm.model import DSSM

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
run_type = hp.run_type
logdir = hp.logdir
batch_size = hp.batch_size
num_epochs = hp.num_epochs
assert hp.run_type in ("new", "continue", "finetune")
if "continue" == hp.run_type:
    load_hparams(hp, logdir)
    batch_size = hp.batch_size
context = Context(hp)

assert hp.train_data is not None
logging.info("# Prepare train/eval batches")
logging.info("Use %s for training set", hp.train_data)
params = {"maxlen1": hp.maxlen1, "maxlen2": hp.maxlen2}
train_batches, num_train_batches, num_train_samples = get_batch(fpath=hp.train_data,
                                                                task_type="set2sca", num_inputfields=2,
                                                                params=params, vocab_fpath=context.vocab,
                                                                batch_size=batch_size, shuffle=True)

# create a iterator of the correct shape and type
iterr = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
x1, x2, score = iterr.get_next()

# 照抄即可，目前不是很熟悉这些接口
train_init_op = iterr.make_initializer(train_batches)

model = DSSM(context)
loss, train_op, global_step, train_summaries = model.train(x1, x2, score)

logging.info("# Session")
saver = tf.train.Saver(max_to_keep=num_epochs)
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    time_sess = time.time()
    ckpt = tf.train.latest_checkpoint(logdir)
    if ckpt is None or "new" == run_type:  # 新建
        save_hparams(hp, logdir)
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
    else:  # continue OR finetune
        saver.restore(sess, ckpt)
        if "finetune" == hp.run_type:  # finetune
            save_hparams(hp, logdir)
        else:  # continue
            batch_size = hp.batch_size

    save_variable_specs(os.path.join(logdir, "var_specs"))
    save_operation_specs(os.path.join(logdir, "op_specs"))
    f_debug = open(os.path.join(logdir, "debug.txt"), "a")
    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    if hp.zero_step:
        sess.run(global_step.assign(0))

    sess.run(train_init_op)
    total_steps = num_epochs * num_train_batches
    logging.info("total_steps:%s, num_epochs:%s, num_train_batches:%s", total_steps, num_epochs, num_train_batches)
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
            model_output = "dssm%02dL%.2f" % (epoch, _loss)
            ckpt_name = os.path.join(logdir, model_output)
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
    # graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=["seq2vec/vec1"])
    # tf.train.write_graph(graph_def, './model', '%s.pb' % model_output, as_text=False)
f_debug.close()
logging.info("Done")
