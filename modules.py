#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from utils import get_shape_list


def ln(inputs, epsilon=1e-8, scope="ln"):
    """Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()  # (N, T_q, C)
        params_shape = inputs_shape[-1:]  # why -1? (C,)
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)  # (N, T_q, 1), (N, T_q, 1)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())  # (C,)
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())  # (C,)
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta  # shift，每个维度有自己的normalization
        
    return outputs


def get_token_embeddings(vocab_size, num_units, zero_pad=True, name="shared_weight_matrix"):
    """Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.

    Returns
    weight variable: (V, E)
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        embeddings = tf.get_variable('weight_mat',
                                     dtype=tf.float32,
                                     shape=(vocab_size, num_units),
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     )
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings


def get_subword_embedding(embedding_table, input_ids):
    # This function assumes that the input is of shape [batch_size, seq_length,
    # num_inputs].
    #
    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])  # [batch_size, seq_length, 1]

    flat_input_ids = tf.reshape(input_ids, [-1])  # [batch_size * seq_length]
    output = tf.gather(embedding_table, flat_input_ids)  # gather是挑选指定索引的值，在这里是一一对应挑选指定的行

    input_shape = get_shape_list(input_ids)

    # [batch_size, seq_length] + [1 * embedding_size]
    embedding_size = get_shape_list(embedding_table)[-1]
    output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * embedding_size])

    return output  # [batch_size, seq_length, embedding_size]


def scaled_dot_product_attention(Q, K, V,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    """See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, Q, K, typee="key")  # (N, T_q, T_k)，此时下面行被pad的为越大负数

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, typee="future")

        # softmax
        outputs = tf.nn.softmax(outputs)  # (N, T_q, T_k)，已成概率值，相当每行query不应该关注后面的key
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # query masking，由于softmax的关系，矩阵下面那些被pad的query又重新有了非0的值，需要mask掉
        outputs = mask(outputs, Q, K, typee="query")

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs


def mask(inputs, queries=None, keys=None, typee=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (N, T_q, T_k)
    queries: 3d tensor. (N, T_q, d)
    keys: 3d tensor. (N, T_k, d)

    e.g.,
    >> queries = tf.constant([[[1.],
                        [2.],
                        [0.]]], tf.float32) # (1, 3, 1)
    >> keys = tf.constant([[[4.],
                     [0.]]], tf.float32)  # (1, 2, 1)
    >> inputs = tf.constant([[[4., 0.],
                               [8., 0.],
                               [0., 0.]]], tf.float32)
    >> mask(inputs, queries, keys, "key")
    array([[[ 4.0000000e+00, -4.2949673e+09],
        [ 8.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09]]], dtype=float32)
    >> inputs = tf.constant([[[1., 0.],
                             [1., 0.],
                              [1., 0.]]], tf.float32)
    >> mask(inputs, queries, keys, "query")
    array([[[1., 0.],
        [1., 0.],
        [0., 0.]]], dtype=float32)
    """
    padding_num = -2 ** 32 + 1
    if typee in ("k", "key", "keys"):
        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)，后面的padding为0
        # 相当于单列(1, 1, ..., 0, 0, 0)转置为单行后平铺为多行
        masks = tf.expand_dims(masks, 1)  # (N, 1, T_k)
        masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)

        # Apply masks to inputs
        paddings = tf.ones_like(inputs) * padding_num  # (N, T_q, T_k)，每个元素为int32最小值
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
    elif typee in ("q", "query", "queries"):
        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)，实际为(1, 1, ..., 0, 0, 0)
        masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)

        # Apply masks to inputs
        outputs = inputs * masks  # 使得下面行数元素为0
    elif typee in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)，左上角为0

        paddings = tf.ones_like(masks) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
        # 相当于左上角全部为超大负数，一softmax就会变成0
        # 对于行i（第i个query），其对于i+1, i+2, ...的权值全部变为0，相当于mask掉future
    else:
        raise Exception("Check if you entered type correctly!")
    return outputs


def multihead_attention(queries, keys, values,
                        num_heads=8, 
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    """Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    """
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(queries, d_model)  # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model)  # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model)  # (N, T_k, d_model)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = ln(outputs)
 
    return outputs

def ff(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3
    
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = ln(outputs)
    
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    V = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)


def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    """Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    """

    E = inputs.get_shape().as_list()[-1]  # static, aka d_model
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices, looks like [[0, 1, 2], [0, 1, 2]]
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

        # First part of the PE function: sin and cos argument, (maxlen, d_model)
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, d_model)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)  # (N, T, d_model)

        # masks
        if masking:  # 把那些没被pad的全部置为位置向量，因为外面是+号，所以要是不mask的话被pad成0那些slot反而会重新有数值
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    """Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    """
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def _dropout_lstm_cell(hparams, train):
    return tf.nn.rnn_cell.DropoutWrapper(
      tf.nn.rnn_cell.LSTMCell(hparams.hidden_size),
      input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))


def lstm(inputs, sequence_length, hparams, train, name, initial_state=None):
    """Adds a stack of LSTM layers on top of input.
    Args:
      inputs: The input `Tensor`, shaped `[batch_size, time_steps, hidden_size]`.
      sequence_length: Lengths of the actual input sequence, excluding padding; a
          `Tensor` shaped `[batch_size]`.
      hparams: HParams; hyperparameters.
      train: bool; `True` when constructing training graph to enable dropout.
      name: string; Create variable names under this scope.
      initial_state: tuple of `LSTMStateTuple`s; the initial state of each layer.
    Returns:
      A tuple (outputs, states), where:
        outputs: The output `Tensor`, shaped `[batch_size, time_steps,
          hidden_size]`.
        states: A tuple of `LSTMStateTuple`s; the final state of each layer.
          Bidirectional LSTM returns a concatenation of last forward and backward
          state, reduced to the original dimensionality.
    """
    layers = [_dropout_lstm_cell(hparams, train)
              for _ in range(hparams.num_hidden_layers)]
    with tf.variable_scope(name):
        return tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.MultiRNNCell(layers),
            inputs,
            sequence_length,
            initial_state=initial_state,
            dtype=tf.float32,
            time_major=False)


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
    """Looks up words embeddings for id tensor.

    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
        ids.
      vocab_size: int. Size of the embedding vocabulary.
      embedding_size: int. Width of the word embeddings.
      initializer_range: float. Embedding initialization range.
      word_embedding_name: string. Name of the embedding table.
      use_one_hot_embeddings: bool. If True, use one-hot method for word
        embeddings. If False, use `tf.gather()`.

    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    # This function assumes that the input is of shape [batch_size, seq_length,
    # num_inputs].
    #
    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])  # [batch_size, seq_length, 1]

    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range))

    flat_input_ids = tf.reshape(input_ids, [-1])  # [batch_size * seq_length]
    if use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.gather(embedding_table, flat_input_ids)  # gather是挑选指定索引的值，在这里是一一对应挑选指定的行

    input_shape = get_shape_list(input_ids)

    # [batch_size, seq_length] + [1 * embedding_size]
    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return output, embedding_table  # [batch_size, seq_length, embedding_size], [vocab_size, embedding_size]

