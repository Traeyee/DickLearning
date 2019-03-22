#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from utils import calc_num_batches
from data_load import encode, load_vocab


def load_data(fpath, maxlen1, maxlen2):
    import re
    ptn = re.compile("\[D:[_A-Z0-9]+\]_")
    sents1, sents2, scores = [], [], []
    with open(fpath, 'r') as f:
        for line in f:
            str_score, sent1, sent2 = line.strip().split("\t")
            spls1 = sent1.split("$$$")
            if 2 != len(spls1):
                continue
            ent1 = [ptn.sub("", _e) for _e in spls1[1].split()]
            sents1.append(" ".join(ent1))
            spls2 = sent2.split("$$$")
            if 2 != len(spls2):
                continue
            ent2 = [ptn.sub("", _e) for _e in spls2[1].split()]
            sents2.append(" ".join(ent2))
            scores.append(float(str_score))
    return sents1, sents2, scores


def generator_fn(sents1, sents2, scores, vocab):
    token2idx, _ = load_vocab(vocab)
    for sent1, sent2, score in zip(sents1, sents2, scores):
        x1 = encode(sent1, "subword", token2idx)
        x2 = encode(sent2, "subword", token2idx)

        yield x1, x2, score


def input_fn(sents1, sents2, scores, vocab_fpath, batch_size, shuffle=False):
    shapes = ([None], [None], ())
    types = (tf.int32, tf.int32, tf.float32)
    paddings = (0, 0, 0.0)

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, scores, vocab_fpath))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle:  # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset


def get_batch(fpath, maxlen1, maxlen2, vocab_fpath, batch_size, shuffle=False):
    """Dssm style task"""
    sents1, sents2, scores = load_data(fpath, maxlen1, maxlen2)
    batches = input_fn(sents1, sents2, scores, vocab_fpath, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size)
    return batches, num_batches, len(sents1)
