#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 22/3/2019 4:05 PM
import json
import sys
import logging
import tensorflow as tf

from instance import CHECK_TASK_TYPE, Instance
from utils import calc_num_batches

assert sys.version_info[0] == 3, "基于Python3"


def CHECK_VOCAB(dict_vocab):
    assert "<pad>" in dict_vocab, "Please add <pad> to your vocabulary manually"
    assert "<unk>" in dict_vocab, "Please add <unk> to your vocabulary manually"
    assert "<s>" in dict_vocab, "Please add <s> to your vocabulary manually"
    assert "</s>" in dict_vocab, "Please add </s> to your vocabulary manually"


def CHECK_VOCAB_DUP(token2idx, idx2token):
    if len(token2idx) != len(idx2token):
        token_set = set()
        dup_list = []
        for token in idx2token.values():
            if token in token_set:
                dup_list.append(token)
            token_set.add(token)
        ord_list = []
        for str1 in dup_list:
            ord_str = "-".join([str(ord(_c)) for _c in str1])
            ord_list.append(ord_str)
        exc_str = "%s != %s, vocab may have duplicate elements: %s\ncode list: %s" % \
                  (len(token2idx), len(idx2token), ", ".join(dup_list), ", ".join(ord_list))
        assert len(token2idx) == len(idx2token), exc_str


def load_vocab(vocab_fpath):
    """Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>

    Returns
    two dictionaries.
    """
    try:
        token2idx = json.loads(vocab_fpath)
        if not isinstance(token2idx, dict):
            raise json.decoder.JSONDecodeError
        CHECK_VOCAB(token2idx)
        return token2idx, None
    except json.decoder.JSONDecodeError:
        vocab = []
        for line in open(vocab_fpath, 'r').read().splitlines():
            w = line.split("\t")[0]
            if w:
                vocab.append(w)
        token2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2token = {idx: token for idx, token in enumerate(vocab)}
        CHECK_VOCAB(token2idx)
        CHECK_VOCAB_DUP(token2idx, idx2token)
        return token2idx, idx2token


def load_data(fpath):
    instances = []
    with open(fpath, 'r') as f:
        for line in f:
            instances.append(line)
    return instances


def input_fn(instances, task_type, input_indices, vocabs, context, batch_size, shuffle=False):
    CHECK_TASK_TYPE(task_type)
    # target
    if task_type.endswith("2sca"):
        target_shape = ()
        target_type = tf.float32
        target_padding = 0.0
    elif task_type.endswith("2cls"):
        target_shape = ()
        target_type = tf.int32
        target_padding = 0
    elif task_type.endswith("2seq"):
        target_shape = [None]
        target_type = tf.int32
        target_padding = 0
    elif task_type.endswith("2vec"):
        target_shape = [None]
        target_type = tf.float32
        target_padding = 0.0
    else:
        assert task_type.endswith("2vecseq")
        target_shape = [None, context.d_imitate]
        target_type = tf.float32
        target_padding = 0.0

    logging.info("温馨提示：如果是seq任务请手动添加<s>, </s>")
    num_input = input_indices.count(",") + 1
    shapes = tuple([[None] for _ in range(num_input)] + [target_shape])
    types = tuple([tf.int32 for _ in range(num_input)] + [target_type])
    paddings = tuple([0 for _ in range(num_input)] + [target_padding])

    # 要从context传递的参数从这里添加, tf.data.Dataset.from_generator不允许传递customized对象
    params = {"maxlens": context.maxlens}

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(instances, task_type, input_indices, vocabs, json.dumps(params)))

    if shuffle:  # for training
        dataset = dataset.shuffle(64 * batch_size)  # TODO: expr

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset

def generator_fn(instances_of_line, task_typee, str_input_indicess, vocabss, str_params):
    def bytes2str(unk):
        ret = unk
        if isinstance(ret, bytes):
            ret = ret.decode("utf-8")
        return ret

    task_type = bytes2str(task_typee)

    str_input_indices = bytes2str(str_input_indicess)
    input_indices = set([int(_idx) for _idx in str_input_indices.split(",")])
    num_input = len(input_indices)

    params = json.loads(bytes2str(str_params))
    accepted_task_type = params.get("accepted_task_type", [])  # TODO: 未实现的接口
    assert type(accepted_task_type) in (list, set, dict)
    maxlens = params.get("maxlens", 0x3f3f)
    if not isinstance(maxlens, list):
        assert isinstance(maxlens, int)
        tmp = [maxlens] * num_input
        maxlens = tmp

    token2idxes = []
    for vocab in bytes2str(vocabss).split(":"):
        token2idx, _ = load_vocab(vocab)
        token2idxes.append(token2idx)

    for line in instances_of_line:
        instance = Instance.load_instance_from_json(line)
        # filter
        if task_type != instance.task_type and instance.task_type not in accepted_task_type:
            tf.logging.info("Instance filtered[task_type=%s]: %s" % (instance.task_type, line))
            continue
        # inputs
        inputs = []
        for i, inputt in enumerate(instance.inputs):
            if i not in input_indices:
                continue
            if len(inputt) > maxlens[i]:
                tf.logging.info("Instance filtered[maxlen_%s]: %s" % (i, line))
                continue
            inputt_idx = [token2idx.get(_t, token2idx["<unk>"]) for _t in inputt]
            if task_type.startswith("set2"):
                inputt_idx = list(set(inputt_idx))
            elif task_type.startswith("seq2"):
                if token2idx["<s>"] != inputt_idx[0]:
                    logging.error("seq2任务input没有以<s>起始")
                if token2idx["</s>"] != inputt_idx[-1]:
                    logging.error("seq2任务input没有以</s>结尾")
            else:
                raise Exception("Wrong task_type")
            inputs.append(inputt_idx)

        # target
        target = instance.target
        if task_type.endswith("2sca"):
            target = float(target)
        elif task_type.endswith("2cls"):
            target = int(target)
        elif task_type.endswith("2vec"):
            target = [float(_t) for _t in target]
        elif task_type.endswith("2vecseq"):
            target = [[float(_v) for _v in _t] for _t in target]
        else:
            assert task_type.endswith("2seq"), "Not implemented yet"
            target_tmp = [token2idx.get(_t, token2idx["<unk>"]) for _t in target]
            if task_type.endswith("2seq"):
                if token2idx["</s>"] != target_tmp[-1]:
                    tf.logging.error("seq2任务target没有以</s>结尾")
            target = target_tmp
        yield tuple(inputs + [target])


def encode(inp, typee, dictt):
    """Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary

    Returns
    list of numbers
    """
    inp_str = inp.decode("utf-8")
    if "subword" == typee:  # Newly added, especially for cnn subword
        tokens = inp_str.split()
        x = [dictt.get(t, dictt["<unk>"]) for t in tokens]
        return x

    elif "x" in typee:
        tokens = inp_str.split() + ["</s>"]
    elif "y" in typee:
        tokens = ["<s>"] + inp_str.split() + ["</s>"]
    else:
        raise Exception("encode wrong type")

    x = [dictt.get(t, dictt["<unk>"]) for t in tokens]
    return x

def get_batch(fpath, task_type, input_indices, vocabs, context, batch_size, shuffle=False):
    """More standarlized, recommended"""
    instances = load_data(fpath)
    batches = input_fn(instances, task_type, input_indices, vocabs, context, batch_size, shuffle)
    num_batches = calc_num_batches(len(instances), batch_size)
    return batches, num_batches, len(instances)
