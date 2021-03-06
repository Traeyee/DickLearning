#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import tensorflow as tf
from utils import calc_num_batches


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
        return token2idx, None
    except json.decoder.JSONDecodeError:
        vocab = [line.split()[0] for line in open(vocab_fpath, 'r').read().splitlines()]
        token2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2token = {idx: token for idx, token in enumerate(vocab)}
        return token2idx, idx2token


def load_data(fpath1, fpath2, maxlen1, maxlen2):
    """Loads source and target data and filters out too lengthy samples.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.

    Returns
    sents1: list of source sents
    sents2: list of target sents
    """
    sents1, sents2 = [], []
    with open(fpath1, 'r') as f1, open(fpath2, 'r') as f2:
        for sent1, sent2 in zip(f1, f2):
            if len(sent1.split()) + 1 > maxlen1:
                continue  # 1: </s>
            if len(sent2.split()) + 1 > maxlen2:
                continue  # 1: </s>
            sents1.append(sent1.strip())
            sents2.append(sent2.strip())
    return sents1, sents2


def load_data2(fpath, maxlen1, maxlen2):
    sents1, sents2, scores = [], [], []
    with open(fpath, 'r') as f:
        for line in f:
            sent1, sent2, str_score = line.strip().split("\t")
            if len(sent1.split()) + 1 > maxlen1:
                continue  # 1: </s>
            if len(sent2.split()) + 1 > maxlen2:
                continue  # 1: </s>
            sents1.append(sent1.strip())
            sents2.append(sent2.strip())
            scores.append(float(str_score))
    return sents1, sents2, scores


def encode(inp, typee, dictt):
    """Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary

    Returns
    list of numbers
    """
    inp_str = inp.decode("utf-8")
    if "x" in typee:
        tokens = inp_str.split() + ["</s>"]
    else:
        tokens = ["<s>"] + inp_str.split() + ["</s>"]

    x = [dictt.get(t, dictt["<unk>"]) for t in tokens]
    return x


def generator_fn(sents1, sents2, vocab):
    """Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.

    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    """
    token2idx, _ = load_vocab(vocab)
    for sent1, sent2 in zip(sents1, sents2):
        x = encode(sent1, "x", token2idx)
        y = encode(sent2, "y", token2idx)
        decoder_input, y = y[:-1], y[1:]

        x_seqlen, y_seqlen = len(x), len(y)
        yield (x, x_seqlen, sent1), (decoder_input, y, y_seqlen, sent2)


def generator_fn_sim(sents1, sents2, scores, vocab):
    token2idx, _ = load_vocab(vocab)
    for sent1, sent2, score in zip(sents1, sents2, scores):
        x1 = encode(sent1, "x1", token2idx)
        x2 = encode(sent2, "x2", token2idx)

        x_seqlen, y_seqlen = len(x1), len(x2)
        yield (x1, x_seqlen, sent1), (x2, y_seqlen, sent2), score


def input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=False):
    """Batchify data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    """
    shapes = (([None], (), ()),
              ([None], [None], (), ()))
    types = ((tf.int32, tf.int32, tf.string),
             (tf.int32, tf.int32, tf.int32, tf.string))
    paddings = ((0, 0, ''),
                (0, 0, 0, ''))

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, vocab_fpath))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle:  # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset


def input_fn_sim(sents1, sents2, scores, vocab_fpath, batch_size, shuffle=False):
    shapes = (([None], (), ()),
              ([None],  (), ()),
              ()
              )
    types = ((tf.int32, tf.int32, tf.string),
             (tf.int32, tf.int32, tf.string),
             tf.float32)
    paddings = ((0, 0, ''),
                (0, 0, ''),
                0.0)

    dataset = tf.data.Dataset.from_generator(
        generator_fn_sim,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, scores, vocab_fpath))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle:  # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset


def get_batch(fpath1, fpath2, maxlen1, maxlen2, vocab_fpath, batch_size, shuffle=False):
    """Gets training / evaluation mini-batches
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    """
    sents1, sents2 = load_data(fpath1, fpath2, maxlen1, maxlen2)
    batches = input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size)
    return batches, num_batches, len(sents1)


def get_batch_sim(fpath, maxlen1, maxlen2, vocab_fpath, batch_size, shuffle=False):
    """Dssm style task"""
    sents1, sents2, scores = load_data2(fpath, maxlen1, maxlen2)
    batches = input_fn_sim(sents1, sents2, scores, vocab_fpath, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size)
    return batches, num_batches, len(sents1)
