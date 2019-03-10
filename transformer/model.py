#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Transformer network
"""
import logging
import tensorflow as tf
from tqdm import tqdm

# from data_load import load_vocab
from transformer.modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, \
    noam_scheme
from transformer.utils import convert_idx_to_token_tensor


class Transformer:
    """
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    """
    def __init__(self, context):
        self.context = context
        # self.token2idx, self.idx2token = load_vocab(hp.vocab)
        self.token2idx, self.idx2token = context.token2idx, context.idx2token
        vocab_size = len(self.token2idx)
        self.embeddings = get_token_embeddings(vocab_size, self.context.d_model, zero_pad=True)

    def encode(self, xs, training=True, name=None):
        """
        Returns
        memory: encoder outputs. (N, T1, d_model)
        """
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x)   # (N, T1, d_model)
            enc *= self.context.d_model**0.5  # scale

            enc += positional_encoding(enc, self.context.maxlen1)
            enc = tf.layers.dropout(enc, self.context.dropout_rate, training=training)

            # # Blocks
            for i in range(self.context.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.context.num_heads,
                                              dropout_rate=self.context.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.context.d_ff, self.context.d_model])
        memory = tf.identity(enc, name=name)
        return memory, sents1

    def decode(self, ys, memory, training=True):
        """
        memory: encoder outputs. (N, T1, d_model)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        """
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys

            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)  # (N, T2, d_model)
            dec *= self.context.d_model ** 0.5  # scale

            dec += positional_encoding(dec, self.context.maxlen2)
            dec = tf.layers.dropout(dec, self.context.dropout_rate, training=training)

            # Blocks
            for i in range(self.context.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              num_heads=self.context.num_heads,
                                              dropout_rate=self.context.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              num_heads=self.context.num_heads,
                                              dropout_rate=self.context.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    # ## Feed Forward
                    dec = ff(dec, num_units=[self.context.d_ff, self.context.d_model])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings)  # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', dec, weights)  # (N, T2, vocab_size)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat, y, sents2

    def train(self, xs, ys):
        """
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        """
        # forward
        memory, sents1 = self.encode(xs)
        logits, preds, y, sents2 = self.decode(ys, memory)

        # train scheme
        y_ = label_smoothing(tf.one_hot(y, depth=len(self.idx2token)))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<pad>"]))  # 0: <pad>
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.context.lr, global_step, self.context.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        """Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        """
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1 = self.encode(xs, False)
        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.context.maxlen2)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]:
                break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()
        return y_hat, summaries

    def debug(self, xs, ys):
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1 = self.encode(xs, False)
        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.context.maxlen2)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]:
                break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)
        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()
        return y_hat, summaries
