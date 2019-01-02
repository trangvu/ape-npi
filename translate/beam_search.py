import tensorflow as tf
from translate import utils


"""
Code from: https://github.com/vahidk/EffectiveTensorflow
"""

def get_weights(sequence, eos_id, include_first_eos=True):
    cumsum = tf.cumsum(tf.to_float(tf.not_equal(sequence, eos_id)), axis=1)
    range_ = tf.range(start=1, limit=tf.shape(sequence)[1] + 1)
    range_ = tf.tile(tf.expand_dims(range_, axis=0), [tf.shape(sequence)[0], 1])
    weights = tf.to_float(tf.equal(cumsum, tf.to_float(range_)))

    if include_first_eos:
        weights = weights[:,:-1]
        shape = [tf.shape(weights)[0], 1]
        weights = tf.concat([tf.ones(tf.stack(shape)), weights], axis=1)

    return tf.stop_gradient(weights)


def resize_like(src, dst):
    batch_size = tf.shape(src)[0]
    beam_size = tf.shape(dst)[0] // batch_size
    shape = get_shape(src)[1:]
    src = tf.tile(tf.expand_dims(src, axis=1), [1, beam_size] + [1] * len(shape))
    src = tf.reshape(src, tf.stack([batch_size * beam_size] + shape))
    return src


def get_shape(tensor):
    """Returns static shape if available and dynamic shape otherwise."""
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims


def batch_gather(tensor, indices):
    """Gather in batch from a tensor of arbitrary size.

    In pseduocode this module will produce the following:
    output[i] = tf.gather(tensor[i], indices[i])

    Args:
      tensor: Tensor of arbitrary size.
      indices: Vector of indices.
    Returns:
      output: A tensor of gathered values.
    """
    shape = get_shape(tensor)
    flat_first = tf.reshape(tensor, [shape[0] * shape[1]] + shape[2:])
    indices = tf.convert_to_tensor(indices)
    offset_shape = [shape[0]] + [1] * (indices.shape.ndims - 1)
    offset = tf.reshape(tf.range(shape[0]) * shape[1], offset_shape)
    output = tf.gather(flat_first, indices + offset)
    return output


def log_softmax(x, axis, temperature=None):
    T = temperature or 1.0
    my_max = tf.reduce_max(x/T, axis=axis, keep_dims=True)
    return x - (tf.log(tf.reduce_sum(tf.exp(x/T - my_max), axis, keep_dims=True)) + my_max)


def rnn_beam_search(update_funs, initial_states, sequence_length, beam_size, len_normalization=None,
                    temperature=None, parallel_iterations=16, swap_memory=True):
    """
    :param update_funs: function to compute the next state and logits given the current state and previous ids
    :param initial_states: recurrent model states
    :param sequence_length: maximum output length
    :param beam_size: beam size
    :param len_normalization: length normalization coefficient (0 or None for no length normalization)
    :return: tensor of size (batch_size, beam_size, seq_len) containing the beam-search hypotheses sorted by
        best score (axis 1), and tensor of size (batch_size, beam_size) containing the said scores.
    """
    batch_size = tf.shape(initial_states[0])[0]

    state_sizes = [tf.shape(state)[1] for state in initial_states]
    states = []
    for initial_state in initial_states:
        state = tf.tile(tf.expand_dims(initial_state, axis=1), [1, beam_size, 1])
        states.append(state)
    states = tf.concat(states, axis=2)

    scores = tf.concat([
        tf.ones(shape=[batch_size, 1]),
        tf.zeros(shape=[batch_size, beam_size - 1])], axis=1)
    scores = tf.log(scores)

    ids = tf.tile([[utils.BOS_ID]], [batch_size, beam_size])
    hypotheses = tf.expand_dims(ids, axis=2)

    mask = tf.ones([batch_size, beam_size], dtype=tf.float32)
    time = tf.constant(0, dtype=tf.int32, name='time')

    def time_step(time, mask, hypotheses, states, token_ids, scores):
        token_ids = tf.reshape(token_ids, [batch_size * beam_size])
        token_scores = tf.zeros([batch_size, beam_size, 1])

        new_states = []
        states = tf.split(states, num_or_size_splits=state_sizes, axis=2)

        for k, (state, state_size, update_fun) in enumerate(zip(states, state_sizes, update_funs)):
            state = tf.reshape(state, [batch_size * beam_size, state_size])

            scope = tf.get_variable_scope() if len(update_funs) == 1 else 'model_{}'.format(k + 1)
            with tf.variable_scope(scope, reuse=True):
                state, logits = update_fun(state, token_ids, time)

            state = tf.reshape(state, [batch_size, beam_size, state_size])
            new_states.append(state)

            num_classes = tf.shape(logits)[1]
            logits = tf.reshape(logits, [batch_size, beam_size, num_classes])
            token_scores += log_softmax(logits, axis=2, temperature=temperature)

        num_classes = tf.shape(token_scores)[2]
        mask1 = tf.expand_dims(mask, axis=2)
        mask2 = tf.one_hot(indices=[[utils.EOS_ID]], depth=num_classes)
        token_scores = token_scores * mask1 + (1 - mask1) * (1 - mask2) * -1e30

        sum_logprobs = tf.expand_dims(scores, axis=2) + token_scores

        scores, indices = tf.nn.top_k(
            tf.reshape(sum_logprobs, [batch_size, num_classes * beam_size]),
            k=beam_size)

        beam_ids = indices // num_classes
        token_ids = indices % num_classes

        states = tf.concat([batch_gather(state, beam_ids) for state in new_states], axis=2)
        hypotheses = tf.concat([batch_gather(hypotheses, beam_ids), tf.expand_dims(token_ids, axis=2)], axis=2)

        mask = (batch_gather(mask, beam_ids) * tf.to_float(tf.not_equal(token_ids, utils.EOS_ID)))
        return time + 1, mask, hypotheses, states, token_ids, scores

    loop_vars = [time, mask, hypotheses, states, ids, scores]
    shapes = [tf.TensorShape([None] * len(var.shape)) for var in loop_vars]

    def cond(time, mask, *_):
        p1 = time < sequence_length
        p2 = tf.to_int32(tf.reduce_sum(1 - mask)) < batch_size * beam_size
        return tf.logical_and(p1, p2)

    _, mask, hypotheses, states, ids, scores = tf.while_loop(
        cond=cond,
        body=time_step,
        loop_vars=loop_vars,
        shape_invariants=shapes,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    hypotheses = hypotheses[:, :, 1:]  # remove BOS symbol

    if len_normalization:
        n = tf.shape(hypotheses)[1]
        sequence_length = tf.shape(hypotheses)[2]
        sel_ids_ = tf.reshape(hypotheses, shape=[batch_size * n, sequence_length])
        mask = get_weights(sel_ids_, utils.EOS_ID, include_first_eos=True)
        length = tf.reduce_sum(mask, axis=1)
        length = tf.reshape(length, shape=[batch_size, n])
        scores /= (length ** len_normalization)
        scores, indices = tf.nn.top_k(scores, k=beam_size, sorted=True)
        indices = tf.stack([tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, beam_size]), indices], axis=2)
        hypotheses = tf.gather_nd(hypotheses, indices)

    return hypotheses, scores
