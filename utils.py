# -*- coding: utf-8 -*-
# /usr/bin/python3
import tensorflow as tf
import json
import os
import re
import logging
import time
import six
from tensorflow.core.framework import device_attributes_pb2
from tensorflow.python import pywrap_tensorflow


def list_local_devices(session_config=None):
    """List the available devices available in the local process.

    Args:
      session_config: a session config proto or None to use the default config.

    Returns:
      A list of `DeviceAttribute` protocol buffers.
    """
    def _convert(pb_str):
        m = device_attributes_pb2.DeviceAttributes()
        m.ParseFromString(pb_str)
        return m

    return [
        _convert(s)
        for s in pywrap_tensorflow.list_devices(session_config=session_config)
    ]


def get_available_gpus():
    local_device_protos = list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def get_gradients_by_loss_and_optimizer(loss, optimizer):
    grads_and_vars = optimizer.compute_gradients(loss)
    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
        raise ValueError(
            "No gradients provided for any variable, check your graph for ops"
            " that do not support gradients, between variables %s and loss %s." %
            ([str(v) for _, v in grads_and_vars], loss))
    return grads_and_vars


def calc_num_batches(total_num, batch_size):
    """Calculates the number of batches.
    total_num: total sample number
    batch_size

    Returns
    number of batches, allowing for remainders."""
    return total_num // batch_size + int(total_num % batch_size != 0)


def convert_idx_to_token_tensor(inputs, idx2token):
    """Converts int32 tensor to string tensor.
    inputs: 1d int32 tensor. indices.
    idx2token: dictionary

    Returns
    1d string tensor.
    """
    def my_func(inputs):
        return " ".join(idx2token[elem] for elem in inputs)

    return tf.py_func(my_func, [inputs], tf.string)

# # def pad(x, maxlen):
# #     """Pads x, list of sequences, and make it as a numpy array.
# #     x: list of sequences. e.g., [[2, 3, 4], [5, 6, 7, 8, 9], ...]
# #     maxlen: scalar
# #
# #     Returns
# #     numpy int32 array of (len(x), maxlen)
# #     """
# #     padded = []
# #     for seq in x:
# #         seq += [0] * (maxlen - len(seq))
# #         padded.append(seq)
# #
# #     arry = np.array(padded, np.int32)
# #     assert arry.shape == (len(x), maxlen), "Failed to make an array"
#
#     return arry


def postprocess(hypotheses, idx2token):
    """Processes translation outputs.
    hypotheses: list of encoded predictions
    idx2token: dictionary

    Returns
    processed hypotheses
    """
    _hypotheses = []
    for h in hypotheses:
        sent = "".join(idx2token[idx] for idx in h)
        sent = sent.split("</s>")[0].strip()
        sent = sent.replace("▁", " ")  # remove bpe symbols
        _hypotheses.append(sent.strip())
    return _hypotheses


def save_hparams(hparams, path):
    """Saves hparams to path
    hparams: argsparse object.
    path: output directory.

    Writes
    hparams as literal dictionary to path.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path, "hparams"), 'w') as fout:
        fout.write(hp)


def load_hparams(hp, path):
    """Loads hparams and overrides parser
    parser: argsparse parser
    path: directory or file where hparams are saved
    """
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    d = open(os.path.join(path, "hparams"), 'r').read()
    flag2val = json.loads(d)
    for f, v in flag2val.items():
        hp.__dict__[f] = v


def save_variable_specs(fpath):
    """Saves information about variables such as
    their name, shape, and total parameter number
    fpath: string. output file path

    Writes
    a text file named fpath.
    """
    def _get_size(shp):
        """Gets size of tensor shape
        shp: TensorShape

        Returns
        size
        """
        size = 1
        for d in range(len(shp)):
            size *= shp[d]
        return size

    params, num_params = [], 0
    for v in tf.global_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    tvs = []
    for tv in tf.trainable_variables():
        tvs.append(tv.name)
    print("num_params: ", num_params)
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
        fout.write("\ntrainable variables:\n")
        fout.write("\n".join(tvs))

    logging.info("Variables info has been saved.")


def save_operation_specs(fpath):
    ops = []
    for op in tf.get_default_graph().get_operations():
        ops.append("{}==={}".format(op.name, op.type))
    with open(fpath, 'w') as fout:
        fout.write("\n".join(ops))
    logging.info("Operaions info has been saved.")


def get_hypotheses(num_batches, num_samples, sess, tensor, dictt, use_profile=False, **kwargs):
    """Gets hypotheses.
    num_batches: scalar.
    num_samples: scalar.
    sess: tensorflow sess object
    tensor: target tensor to fetch
    dict: idx2token dictionary

    Returns
    hypotheses: list of sents
    """
    hypotheses = []
    for ith_batch in range(num_batches):
        ts = time.time()
        if use_profile:
            run_metadata = kwargs['run_metadata']
            h = sess.run(tensor, options=kwargs['options'], run_metadata=run_metadata)
            kwargs['profiler'].add_step(step=ith_batch, run_meta=run_metadata)
        else:
            h = sess.run(tensor)
        logging.info("get_hypotheses %sth: %s ", ith_batch, (time.time() - ts))
        hypotheses.extend(h.tolist())
    hypotheses = postprocess(hypotheses, dictt)

    return hypotheses[:num_samples]


def calc_bleu(ref, translation):
    """Calculates bleu score and appends the report to translation
    ref: reference file path
    translation: model output file path

    Returns
    translation that the bleu score is appended to"""
    get_bleu_score = "perl multi-bleu.perl {} < {} > {}".format(ref, translation, "temp")
    os.system(get_bleu_score)
    bleu_score_report = open("temp", "r").read()
    with open(translation, "a") as fout:
        fout.write("\n{}".format(bleu_score_report))
    try:
        score = re.findall("BLEU = ([^,]+)", bleu_score_report)[0]
        new_translation = translation + "B{}".format(score)
        os.system("mv {} {}".format(translation, new_translation))
        os.remove(translation)

    except:
        pass
    os.remove("temp")


# def get_inference_variables(ckpt, filter):
#     reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
#     var_to_shape_map = reader.get_variable_to_shape_map()
#     vars = [v for v in sorted(var_to_shape_map) if filter not in v]
#     return vars

def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))



def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape
