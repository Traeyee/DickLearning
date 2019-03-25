#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 22/3/2019 4:05 PM
import json
import sys
import tensorflow as tf

from instance import Instance, CHECK_TASK_TYPE, Instance2
from utils import calc_num_batches

assert sys.version_info[0] == 3, "基于Python3"


def CHECK_VOCAB(dict_vocab):
    assert "<pad>" in dict_vocab, "Please add <pad> to your vocabulary manually"
    assert "<unk>" in dict_vocab, "Please add <unk> to your vocabulary manually"
    assert "<s>" in dict_vocab, "Please add <s> to your vocabulary manually"
    assert "</s>" in dict_vocab, "Please add </s> to your vocabulary manually"


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
        vocab = [line.split("\t")[0] for line in open(vocab_fpath, 'r').read().splitlines()]
        token2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2token = {idx: token for idx, token in enumerate(vocab)}
        CHECK_VOCAB(token2idx)
        return token2idx, idx2token


def load_data(fpath):
    instances = []
    with open(fpath, 'r') as f:
        for line in f:
            instances.append(line)
    return instances


def input_fn(instances, task_type, num_inputfields, params, vocab_fpath, batch_size, shuffle=False):
    CHECK_TASK_TYPE(task_type)
    if task_type.endswith("2sca"):
        target_shape = ()
        target_type = tf.float32
        target_padding = 0.0
    elif task_type.endswith("2cls"):
        target_shape = ()
        target_type = tf.int32
        target_padding = 0
    else:
        target_shape = [None]
        target_type = tf.int32
        target_padding = 0

    if 2 == num_inputfields or task_type.endswith("2seq"):
        shapes = ([None], [None], target_shape)
        types = (tf.int32, tf.int32, target_type)
        paddings = (0, 0, target_padding)
    else:  # 2
        shapes = ([None], target_shape)
        types = (tf.int32, target_type)
        paddings = (0, target_padding)
    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(instances, task_type, num_inputfields, json.dumps(params), vocab_fpath))

    if shuffle:  # for training
        dataset = dataset.shuffle(2 * batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset


def input_fn2(instances, task_type, input_indices, vocabs, params, batch_size, shuffle=False):
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
    else:
        target_shape = [None]
        target_type = tf.int32
        target_padding = 0

    tf.logging.info("温馨提示：如果是2seq任务请手动添加<s>, </s>")
    num_input = input_indices.count(",") + 1
    shapes = tuple([[None] for _ in range(num_input)] + [target_shape])
    types = tuple([tf.int32 for _ in range(num_input)] + [target_type])
    paddings = tuple([0 for _ in range(num_input)] + [target_padding])
    dataset = tf.data.Dataset.from_generator(
        generator_fn2,
        output_shapes=shapes,
        output_types=types,
        args=(instances, task_type, input_indices, vocabs, json.dumps(params)))

    if shuffle:  # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset

def generator_fn2(instances_of_line, task_typee, str_input_indicess, vocabss, str_params):
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
    accepted_task_type = params.get("accepted_task_type", [])
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
        instance = Instance2.load_instance_from_json(line)
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
                if token2idx["</s>"] != inputt_idx[-1]:
                    tf.logging.error("seq2任务input没有以</s>结尾")
            else:
                raise Exception("Wrong task_type")
            inputs.append(inputt_idx)

        # target
        target = instance.target
        if task_type.endswith("2sca"):
            target = float(target)
        elif task_type.endswith("2cls"):
            target = int(target)
        else:
            target_tmp = [token2idx.get(_t, token2idx["<unk>"]) for _t in target]
            if task_type.endswith("2seq"):
                if token2idx["</s>"] != target_tmp[-1]:
                    tf.logging.error("seq2任务target没有以</s>结尾")
            target = target_tmp
        yield tuple(inputs + [target])


def generator_fn(instances_of_line, task_typee, num_inputfields, str_params, vocab):
    task_type = task_typee
    if isinstance(task_type, bytes):
        task_type = task_type.decode("utf-8")
    params = json.loads(str_params)
    accepted_task_type = params.get("accepted_task_type", [])
    assert type(accepted_task_type) in (list, set, dict)
    maxlen1 = params.get("maxlen1", 0x3f3f)
    maxlen2 = params.get("maxlen2", 0x3f3f)

    assert num_inputfields in (1, 2)
    token2idx, _ = load_vocab(vocab)

    for line in instances_of_line:
        instance = Instance.load_instance_from_json(line)
        # filter
        if task_type != instance.task_type and instance.task_type not in accepted_task_type:
            continue
        if len(instance.input1) + 1 > maxlen1:
            continue  # 1: </s>
        if 2 == num_inputfields:
            if not instance.input2:
                continue
            if len(instance.input2) + 1 > maxlen2:
                continue  # 1: </s>

        input1, target = instance.input1, instance.target
        # input1
        input1 = [token2idx.get(_t, token2idx["<unk>"]) for _t in input1]
        if task_type.startswith("set2"):
            input1 = list(set(input1))
        elif task_type.startswith("seq2"):
            input1 = input1 + [token2idx["</s>"]]
        else:
            raise Exception("Wrong task_type")
        # target
        if task_type.endswith("2sca"):
            target = float(target)
        elif task_type.endswith("2cls"):
            target = int(target)
        else:
            target_tmp = [token2idx.get(_t, token2idx["<unk>"]) for _t in target]
            if task_type.endswith("2seq"):
                assert 1 == num_inputfields
                input2 = [token2idx["<s>"]] + target_tmp
                target = target_tmp + [token2idx["</s>"]]
                yield input1, input2, target
            else:
                raise Exception("Wrong task_type")

        if 1 == num_inputfields:
            yield input1, target
        else:
            # input2
            input2 = instance.input2
            input2 = [token2idx.get(_t, token2idx["<unk>"]) for _t in input2]
            if task_type.startswith("set2"):
                input2 = list(set(input2))
            elif task_type.startswith("seq2"):
                input2 = input2 + [token2idx["</s>"]]
            else:
                raise Exception("Wrong task_type")
            yield input1, input2, target


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


def get_batch(fpath, task_type, num_inputfields, params, vocab_fpath, batch_size, shuffle=False):
    """More standarlized, recommended"""
    instances = load_data(fpath)
    batches = input_fn(instances, task_type, num_inputfields, params, vocab_fpath, batch_size, shuffle)
    num_batches = calc_num_batches(len(instances), batch_size)
    return batches, num_batches, len(instances)


def get_batch2(fpath, task_type, input_indices, vocabs, params, batch_size, shuffle=False):
    """More standarlized, recommended"""
    instances = load_data(fpath)
    batches = input_fn2(instances, task_type, input_indices, vocabs, params, batch_size, shuffle)
    num_batches = calc_num_batches(len(instances), batch_size)
    return batches, num_batches, len(instances)
