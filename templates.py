#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 24 March 2019 21:03
import logging
import os
import time
import math
import tensorflow as tf
from tqdm import tqdm

from context import Context
from hparams import Hparams
from utils import save_hparams, save_variable_specs, save_operation_specs, load_hparams
from data_load import get_batch


def train_template(class_model, shuffle=True, save_model=True):  # 大数据集耗时请关掉shuffle，调参请关掉save_model
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
    task_type = hp.task_type
    assert hp.run_type in ("new", "continue", "finetune")
    if "continue" == hp.run_type:
        load_hparams(hp, logdir)
        batch_size = hp.batch_size
        if task_type is not None:
            assert task_type == hp.task_type
        task_type = hp.task_type
    assert task_type is not None
    context = Context(hp)
    logging.info("# Prepare train/eval batches")
    logging.info("Use %s for training set", hp.train_data)
    logging.info("Use %s for evaluation set", hp.eval_data)
    eval_batches, num_eval_batches, num_eval_samples = get_batch(fpath=hp.eval_data,
                                                                 task_type=task_type,
                                                                 input_indices=context.input_indices,
                                                                 vocabs=context.vocabs,
                                                                 context=context,
                                                                 batch_size=batch_size, shuffle=False)
    train_batches, num_train_batches, num_train_samples = get_batch(fpath=hp.train_data,
                                                                    task_type=task_type,
                                                                    input_indices=context.input_indices,
                                                                    vocabs=context.vocabs,
                                                                    context=context,
                                                                    batch_size=batch_size, shuffle=shuffle)

    # create a iterator of the correct shape and type
    iterr = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
    inputs_and_target = iterr.get_next()

    # 照抄即可，目前不是很熟悉这些接口
    train_init_op = iterr.make_initializer(train_batches)
    eval_init_op = iterr.make_initializer(eval_batches)
    model = class_model(context)
    loss, train_op, global_step, train_summaries = model.train(inputs=inputs_and_target[:-1], targets=inputs_and_target[-1])
    eval_ouputs, eval_summaries = model.eval(inputs=inputs_and_target[:-1], targets=inputs_and_target[-1])
    inference_name = model.get_inference_op_name()
    logging.info("inference_node_name:%s" % inference_name)

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
            f_debug.write("train: epoch %s takes %s\n" % (epoch, time.time() - ts))
            summary_writer.add_summary(_summary, _gs)

            if _gs and _gs % num_train_batches == 0:
                logging.info("epoch {} is done".format(epoch))

                # train loss
                _loss = sess.run(loss)
                # eval
                logging.info("# eval evaluation")
                _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
                summary_writer.add_summary(_eval_summaries, _gs)
                if save_model:
                    # save checkpoint
                    logging.info("# save models")
                    model_output = "model%02dL%.2f" % (epoch, _loss)
                    ckpt_name = os.path.join(logdir, model_output)
                    saver.save(sess, ckpt_name, global_step=_gs)
                    logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))
                # proceed to next epoch
                logging.info("# fall back to train mode")
                ts = time.time()
                sess.run(train_init_op)
                logging.info("fallback_train: %s\t%s\t%s takes %s" % (i, _gs, epoch, time.time() - ts))
                logging.info("epoch %s takes %s", epoch, time.time() - t_epoch)
                t_epoch = time.time()
        summary_writer.close()
        logging.info("Session runs for %s", time.time() - time_sess)
        if save_model:
            # save to pb
            inference_node_name = inference_name[:inference_name.find(":")]
            graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                     output_node_names=[inference_node_name])
            tf.train.write_graph(graph_def, logdir, '%s.pb' % model_output, as_text=False)
    f_debug.close()
    logging.info("Done")


def export_pb_template(class_model):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    logging.info("# hparams")
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    load_hparams(hp, hp.logdir)
    context = Context(hp)

    params = {"maxlens": 0x3f3f}
    eval_batches, num_eval_batches, num_eval_samples = get_batch(fpath=hp.eval_data,
                                                                 task_type=hp.task_type,
                                                                 input_indices=context.input_indices,
                                                                 vocabs=context.vocabs,
                                                                 context=params,
                                                                 batch_size=hp.batch_size, shuffle=True)

    # create a iterator of the correct shape and type
    iterr = tf.data.Iterator.from_structure(eval_batches.output_types, eval_batches.output_shapes)
    inputs_and_target = iterr.get_next()

    model = class_model(context)
    _ = model.eval(inputs_and_target[:-1], inputs_and_target[-1])
    inference_name = model.get_inference_op_name()
    logging.info("inference_node_name:%s" % inference_name)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(hp.logdir)
        saver.restore(sess, ckpt)
        inference_node_name = inference_name[:inference_name.find(":")]
        graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                 output_node_names=[inference_node_name])
        tf.train.write_graph(graph_def, './model', '%s.pb' % hp.pb_name, as_text=False)
        save_operation_specs(os.path.join("./model", '%s.ops' % hp.pb_name))
