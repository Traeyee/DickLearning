#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 4/3/2019 11:32 AM
import os
import sys
import logging
import tensorflow as tf
import numpy as np

from transformer.data_load import encode
from nlptools import get_index_dict

logger = logging.getLogger()
logger.setLevel(logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


token2idx, _ = get_index_dict()


def get_embedding(session, sent):
    idx = encode(" ".join(sent).encode("utf-8"), "x", token2idx)
    arr = np.array([idx])
    input_x = session.graph.get_tensor_by_name("encoder/input_x:0")
    vec1 = session.graph.get_tensor_by_name("seq2vec/vec:0")
    outputs = session.run(fetches=[vec1], feed_dict={input_x: arr})
    return outputs[0][0]


def main():
    with tf.Session() as restore_sess:
        graph_def = tf.GraphDef()
        with open("./model/default.pb", "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        line = "中国军队||中国人民解放军"
        print("input:")
        for line in sys.stdin:
            sents = line.strip().split("||")
            assert 2 == len(sents)
            vec1 = get_embedding(restore_sess, sents[0])
            vec2 = get_embedding(restore_sess, sents[1])
            vec1_l2 = np.sqrt(np.sum(np.square(vec1)))
            vec2_l2 = np.sqrt(np.sum(np.square(vec2)))
            sim = np.sum(vec1 * vec2) / (vec1_l2 * vec2_l2)
            print(sim)
            print("input:")


if __name__ == '__main__':
    main()
