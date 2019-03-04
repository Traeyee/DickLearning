#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 1/3/2019 11:39 AM
import logging
import os
import math
import time
import tensorflow as tf
from tqdm import tqdm

from transformer.hparams import Hparams
from transformer.model import Transformer
from transformer.utils import save_hparams, save_variable_specs, get_hypotheses
from transformer.data_load import get_batch

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)

logging.info("# Prepare train/eval batches")
train_batches, num_train_batches, num_train_samples = get_batch(hp.train1, hp.train2, hp.maxlen1, hp.maxlen2,
                                                                hp.vocab, hp.batch_size, shuffle=True)
eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval1, hp.eval2, 100000, 100000,
                                                             hp.vocab, hp.eval_batch_size, shuffle=False)

# create a iterator of the correct shape and type
iterr = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys = iterr.get_next()

# 照抄即可，目前不是很熟悉这些接口
train_init_op = iterr.make_initializer(train_batches)
eval_init_op = iterr.make_initializer(eval_batches)

logging.info("# Load model")

m = Transformer(hp)
loss, train_op, global_step, train_summaries = m.train(xs, ys)
y_hat, eval_summaries = m.eval(xs, ys)
# y_hat = m.infer(xs, ys)

logging.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)
    for i in tqdm(range(_gs, total_steps + 1)):
        ts = time.time()
        _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
        epoch = math.ceil(_gs / num_train_batches)
        logging.info("train: %s\t%s\t%s takes %s" % (i, _gs, epoch, time.time() - ts))
        summary_writer.add_summary(_summary, _gs)

        if _gs and _gs % num_train_batches == 0:
            logging.info("epoch {} is done".format(epoch))
            _loss = sess.run(loss)  # train loss

            logging.info("# test evaluation")
            _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
            summary_writer.add_summary(_eval_summaries, _gs)

            logging.info("# get hypotheses")
            ts = time.time()
            hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.idx2token)
            logging.info("eval: %s\t%s\t%s takes %s" % (i, _gs, epoch, time.time() - ts))

            logging.info("# write results")
            model_output = "couplet%02dL%.2f" % (epoch, _loss)
            if not os.path.exists(hp.evaldir):
                os.makedirs(hp.evaldir)
            translation = os.path.join(hp.evaldir, model_output)
            with open(translation, 'w') as fout:
                fout.write("\n".join(hypotheses))

            logging.info("# calc bleu score and append it to translation")
            # calc_bleu(hp.eval3, translation)

            logging.info("# save models")
            ckpt_name = os.path.join(hp.logdir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# fall back to train mode")
            ts = time.time()
            sess.run(train_init_op)
            logging.info("fallback_train: %s\t%s\t%s takes %s" % (i, _gs, epoch, time.time() - ts))
    summary_writer.close()


logging.info("Done")
