import tensorflow as tf
import math
from tensorflow.contrib.rnn import BasicLSTMCell, RNNCell, DropoutWrapper, MultiRNNCell
from translate.rnn import stack_bidirectional_dynamic_rnn, CellInitializer, GRUCell, DropoutGRUCell, PLSTM
from translate.rnn import get_state_size
from translate.beam_search import get_weights
from translate import utils, beam_search
from translate.conv_lstm import BasicConvLSTMCell


def auto_reuse(fun):
    """
    Wrapper that automatically handles the `reuse' parameter.
    This is rather risky, as it can lead to reusing variables
    by mistake.
    """

    def fun_(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except ValueError as e:
            if 'reuse' in str(e):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    return fun(*args, **kwargs)
            else:
                raise e

    return fun_


get_variable = auto_reuse(tf.get_variable)
dense = auto_reuse(tf.layers.dense)


class CellWrapper(RNNCell):
    """
    Wrapper around LayerNormBasicLSTMCell, BasicLSTMCell and MultiRNNCell, to keep
    the state_is_tuple=False behavior (soon to be deprecated).
    """

    def __init__(self, cell):
        super(CellWrapper, self).__init__()
        self.cell = cell
        self.num_splits = len(cell.state_size) if isinstance(cell.state_size, tuple) else 1

    @property
    def state_size(self):
        return sum(self.cell.state_size)

    @property
    def output_size(self):
        return self.cell.output_size

    def __call__(self, inputs, state, scope=None):
        state = tf.split(value=state, num_or_size_splits=self.num_splits, axis=1)
        new_h, new_state = self.cell(inputs, state, scope=scope)
        return new_h, tf.concat(new_state, 1)


def multi_encoder(encoder_inputs, encoders, encoder_input_length, other_inputs=None, training=True, **kwargs):
    """
    Build multiple encoders according to the configuration in `encoders`, reading from `encoder_inputs`.
    The result is a list of the outputs produced by those encoders (for each time-step), and their final state.

    :param encoder_inputs: list of tensors of shape (batch_size, input_length), one tensor for each encoder.
    :param encoders: list of encoder configurations
    :param encoder_input_length: list of tensors of shape (batch_size,) (one tensor for each encoder)
    :return:
      encoder outputs: a list of tensors of shape (batch_size, input_length, encoder_cell_size), hidden states of the
        encoders.
      encoder state: concatenation of the final states of all encoders, tensor of shape (batch_size, sum_of_state_sizes)
      new_encoder_input_length: list of tensors of shape (batch_size,) with the true length of the encoder outputs.
        May be different than `encoder_input_length` because of maxout strides, and time pooling.
    """
    encoder_states = []
    encoder_outputs = []
    new_encoder_input_length = []

    for i, encoder in enumerate(encoders):

        # create embeddings in the global scope (allows sharing between encoder and decoder)
        weight_scale = encoder.embedding_weight_scale or encoder.weight_scale
        if weight_scale is None:
            initializer = None  # FIXME
        elif encoder.embedding_initializer == 'uniform' or (encoder.embedding_initializer is None
                                                            and encoder.initializer == 'uniform'):
            initializer = tf.random_uniform_initializer(minval=-weight_scale, maxval=weight_scale)
        else:
            initializer = tf.random_normal_initializer(stddev=weight_scale)

        with tf.device('/cpu:0'):  # embeddings can take a very large amount of memory, so
            # storing them in GPU memory can be impractical
            if encoder.binary:
                embeddings = None  # inputs are token ids, which need to be mapped to vectors (embeddings)
            else:
                embedding_shape = [encoder.vocab_size, encoder.embedding_size]
                embeddings = get_variable('embedding_{}'.format(encoder.name), shape=embedding_shape,
                                          initializer=initializer)
            if encoder.pos_embedding_size:
                pos_embedding_shape = [encoder.max_len + 1, encoder.pos_embedding_size]
                pos_embeddings = get_variable('pos_embedding_{}'.format(encoder.name), shape=pos_embedding_shape,
                                              initializer=initializer)
            else:
                pos_embeddings = None

        if encoder.use_lstm is False:
            encoder.cell_type = 'GRU'

        cell_output_size, cell_state_size = get_state_size(encoder.cell_type, encoder.cell_size,
                                                           encoder.lstm_proj_size)

        with tf.variable_scope('encoder_{}'.format(encoder.name)):
            encoder_inputs_ = encoder_inputs[i]
            initial_inputs = encoder_inputs_
            encoder_input_length_ = encoder_input_length[i]

            def get_cell(input_size=None, reuse=False):
                if encoder.cell_type.lower() == 'lstm':
                    cell = CellWrapper(BasicLSTMCell(encoder.cell_size, reuse=reuse))
                elif encoder.cell_type.lower() == 'plstm':
                    cell = PLSTM(encoder.cell_size, reuse=reuse, fact_size=encoder.lstm_fact_size,
                                 proj_size=encoder.lstm_proj_size)
                elif encoder.cell_type.lower() == 'dropoutgru':
                    cell = DropoutGRUCell(encoder.cell_size, reuse=reuse, layer_norm=encoder.layer_norm,
                                          input_size=input_size, input_keep_prob=encoder.rnn_input_keep_prob,
                                          state_keep_prob=encoder.rnn_state_keep_prob)
                else:
                    cell = GRUCell(encoder.cell_size, reuse=reuse, layer_norm=encoder.layer_norm)

                if encoder.use_dropout and encoder.cell_type.lower() != 'dropoutgru':
                    cell = DropoutWrapper(cell, input_keep_prob=encoder.rnn_input_keep_prob,
                                          output_keep_prob=encoder.rnn_output_keep_prob,
                                          state_keep_prob=encoder.rnn_state_keep_prob,
                                          variational_recurrent=encoder.pervasive_dropout,
                                          dtype=tf.float32, input_size=input_size)
                return cell

            batch_size = tf.shape(encoder_inputs_)[0]
            time_steps = tf.shape(encoder_inputs_)[1]

            if embeddings is not None:
                flat_inputs = tf.reshape(encoder_inputs_, [tf.multiply(batch_size, time_steps)])
                flat_inputs = tf.nn.embedding_lookup(embeddings, flat_inputs)
                encoder_inputs_ = tf.reshape(flat_inputs,
                                             tf.stack([batch_size, time_steps, flat_inputs.get_shape()[1].value]))
            if pos_embeddings is not None:
                pos_inputs_ = tf.range(time_steps, dtype=tf.int32)
                pos_inputs_ = tf.nn.embedding_lookup(pos_embeddings, pos_inputs_)
                pos_inputs_ = tf.tile(tf.expand_dims(pos_inputs_, axis=0), [batch_size, 1, 1])
                encoder_inputs_ = tf.concat([encoder_inputs_, pos_inputs_], axis=2)

            if other_inputs is not None:
                encoder_inputs_ = tf.concat([encoder_inputs_, other_inputs], axis=2)

            if encoder.use_dropout:
                noise_shape = [1, time_steps, 1] if encoder.pervasive_dropout else [batch_size, time_steps, 1]
                encoder_inputs_ = tf.nn.dropout(encoder_inputs_, keep_prob=encoder.word_keep_prob,
                                                noise_shape=noise_shape)

                size = tf.shape(encoder_inputs_)[2]
                noise_shape = [1, 1, size] if encoder.pervasive_dropout else [batch_size, time_steps, size]
                encoder_inputs_ = tf.nn.dropout(encoder_inputs_, keep_prob=encoder.embedding_keep_prob,
                                                noise_shape=noise_shape)

            if encoder.input_layers:
                for j, layer_size in enumerate(encoder.input_layers):
                    if encoder.input_layer_activation is not None and encoder.input_layer_activation.lower() == 'relu':
                        activation = tf.nn.relu
                    else:
                        activation = tf.tanh

                    if encoder.batch_norm:
                        encoder_inputs_ = tf.layers.batch_normalization(encoder_inputs_, training=training,
                                                                        name='input_batch_norm_{}'.format(j + 1))

                    encoder_inputs_ = dense(encoder_inputs_, layer_size, activation=activation, use_bias=True,
                                            name='layer_{}'.format(j))
                    if encoder.use_dropout:
                        encoder_inputs_ = tf.nn.dropout(encoder_inputs_, keep_prob=encoder.input_layer_keep_prob)

            if encoder.conv_filters:
                encoder_inputs_ = tf.expand_dims(encoder_inputs_, axis=3)

                for k, out_channels in enumerate(encoder.conv_filters, 1):
                    in_channels = encoder_inputs_.get_shape()[-1].value
                    filter_height, filter_width = encoder.conv_size

                    strides = encoder.conv_strides or [1, 1]
                    strides = [1] + strides + [1]

                    filter_ = get_variable('filter_{}'.format(k),
                                           [filter_height, filter_width, in_channels, out_channels])
                    encoder_inputs_ = tf.nn.conv2d(encoder_inputs_, filter_, strides, padding='SAME')

                    if encoder.batch_norm:
                        encoder_inputs_ = tf.layers.batch_normalization(encoder_inputs_, training=training,
                                                                        name='conv_batch_norm_{}'.format(k))
                    if encoder.conv_activation is not None and encoder.conv_activation.lower() == 'relu':
                        encoder_inputs_ = tf.nn.relu(encoder_inputs_)

                    encoder_input_length_ = tf.to_int32(tf.ceil(encoder_input_length_ / strides[1]))

                feature_size = encoder_inputs_.shape[2].value
                channels = encoder_inputs_.shape[3].value
                time_steps = tf.shape(encoder_inputs_)[1]

                encoder_inputs_ = tf.reshape(encoder_inputs_, [batch_size, time_steps, feature_size * channels])
                conv_outputs_ = encoder_inputs_

                if encoder.conv_lstm_size:
                    cell = BasicConvLSTMCell([feature_size, channels], encoder.conv_lstm_size, 1)
                    encoder_inputs_, _ = tf.nn.bidirectional_dynamic_rnn(
                        cell, cell, encoder_inputs_,
                        dtype=tf.float32
                    )
                    encoder_inputs_ = tf.concat(encoder_inputs_, axis=2)

            if encoder.convolutions:
                if encoder.binary:
                    raise NotImplementedError

                pad = tf.nn.embedding_lookup(embeddings, utils.BOS_ID)
                pad = tf.expand_dims(tf.expand_dims(pad, axis=0), axis=1)
                pad = tf.tile(pad, [batch_size, 1, 1])

                # Fully Character-Level NMT without Explicit Segmentation, Lee et al. 2016
                inputs = []

                for w, filter_size in enumerate(encoder.convolutions, 1):
                    filter_ = get_variable('filter_{}'.format(w), [w, encoder.embedding_size, filter_size])

                    if w > 1:
                        right = (w - 1) // 2
                        left = (w - 1) - right
                        pad_right = tf.tile(pad, [1, right, 1])
                        pad_left = tf.tile(pad, [1, left, 1])
                        inputs_ = tf.concat([pad_left, encoder_inputs_, pad_right], axis=1)
                    else:
                        inputs_ = encoder_inputs_

                    inputs_ = tf.nn.convolution(inputs_, filter=filter_, padding='VALID')
                    inputs.append(inputs_)

                encoder_inputs_ = tf.concat(inputs, axis=2)
                # if encoder.convolution_activation.lower() == 'relu':
                encoder_inputs_ = tf.nn.relu(encoder_inputs_)

            if encoder.maxout_stride:
                if encoder.binary:
                    raise NotImplementedError

                stride = encoder.maxout_stride
                k = tf.to_int32(tf.ceil(time_steps / stride) * stride) - time_steps  # TODO: simpler
                pad = tf.zeros([batch_size, k, tf.shape(encoder_inputs_)[2]])
                encoder_inputs_ = tf.concat([encoder_inputs_, pad], axis=1)
                encoder_inputs_ = tf.nn.pool(encoder_inputs_, window_shape=[stride], pooling_type='MAX',
                                             padding='VALID', strides=[stride])
                encoder_input_length_ = tf.to_int32(tf.ceil(encoder_input_length_ / stride))

            if encoder.highway_layers:
                x = encoder_inputs_
                for j in range(encoder.highway_layers):
                    size = x.shape[2].value

                    with tf.variable_scope('highway_{}'.format(j + 1)):
                        g = tf.layers.dense(x, size, activation=tf.nn.sigmoid, use_bias=True, name='g')
                        y = tf.layers.dense(x, size, activation=tf.nn.relu, use_bias=True, name='y')
                        x = g * y + (1 - g) * x

                encoder_inputs_ = x

            # Contrary to Theano's RNN implementation, states after the sequence length are zero
            # (while Theano repeats last state)
            inter_layer_keep_prob = None if not encoder.use_dropout else encoder.inter_layer_keep_prob

            parameters = dict(
                inputs=encoder_inputs_, sequence_length=encoder_input_length_,
                dtype=tf.float32, parallel_iterations=encoder.parallel_iterations,
                inter_layers=encoder.inter_layers, inter_layer_activation=encoder.inter_layer_activation,
                batch_norm=encoder.batch_norm, inter_layer_keep_prob=inter_layer_keep_prob,
                pervasive_dropout=encoder.pervasive_dropout, training=training
            )

            input_size = encoder_inputs_.get_shape()[2].value

            def get_initial_state(name='initial_state'):
                if encoder.train_initial_states:
                    initial_state = get_variable(name, initializer=tf.zeros(cell_state_size))
                    return tf.tile(tf.expand_dims(initial_state, axis=0), [batch_size, 1])
                else:
                    return None

            if encoder.bidir:
                rnn = lambda reuse: stack_bidirectional_dynamic_rnn(
                    cells_fw=[get_cell(input_size if j == 0 else 2 * cell_output_size, reuse=reuse)
                              for j in range(encoder.layers)],
                    cells_bw=[get_cell(input_size if j == 0 else 2 * cell_output_size, reuse=reuse)
                              for j in range(encoder.layers)],
                    initial_states_fw=[get_initial_state('initial_state_fw')] * encoder.layers,
                    initial_states_bw=[get_initial_state('initial_state_bw')] * encoder.layers,
                    time_pooling=encoder.time_pooling, pooling_avg=encoder.pooling_avg,
                    **parameters)

                initializer = CellInitializer(encoder.cell_size) if encoder.orthogonal_init else None
                with tf.variable_scope(tf.get_variable_scope(), initializer=initializer):
                    try:
                        encoder_outputs_, _, encoder_states_ = rnn(reuse=False)
                    except ValueError:  # Multi-task scenario where we're reusing the same RNN parameters
                        encoder_outputs_, _, encoder_states_ = rnn(reuse=True)
            else:
                if encoder.time_pooling or encoder.final_state == 'concat_last':
                    raise NotImplementedError

                if encoder.layers > 1:
                    cell = MultiRNNCell([get_cell(input_size if j == 0 else cell_output_size)
                                         for j in range(encoder.layers)])
                    initial_state = (get_initial_state(),) * encoder.layers
                else:
                    cell = get_cell(input_size)
                    initial_state = get_initial_state()

                encoder_outputs_, encoder_states_ = auto_reuse(tf.nn.dynamic_rnn)(cell=cell,
                                                                                  initial_state=initial_state,
                                                                                  **parameters)

            if encoder.time_pooling:
                for stride in encoder.time_pooling[:encoder.layers - 1]:
                    encoder_input_length_ = (encoder_input_length_ + stride - 1) // stride  # rounding up

            last_backward = encoder_outputs_[:, 0, cell_output_size:]
            indices = tf.stack([tf.range(batch_size), encoder_input_length_ - 1], axis=1)
            last_forward = tf.gather_nd(encoder_outputs_[:, :, :cell_output_size], indices)
            last_forward.set_shape([None, cell_output_size])

            if encoder.final_state == 'concat_last':  # concats last states of all backward layers (full LSTM states)
                encoder_state_ = tf.concat(encoder_states_, axis=1)
            elif encoder.final_state == 'average':
                mask = tf.sequence_mask(encoder_input_length_, maxlen=tf.shape(encoder_outputs_)[1], dtype=tf.float32)
                mask = tf.expand_dims(mask, axis=2)
                encoder_state_ = tf.reduce_sum(mask * encoder_outputs_, axis=1) / tf.reduce_sum(mask, axis=1)
            elif encoder.final_state == 'average_inputs':
                mask = tf.sequence_mask(encoder_input_length_, maxlen=tf.shape(encoder_inputs_)[1], dtype=tf.float32)
                mask = tf.expand_dims(mask, axis=2)
                encoder_state_ = tf.reduce_sum(mask * encoder_inputs_, axis=1) / tf.reduce_sum(mask, axis=1)
            elif encoder.bidir and encoder.final_state == 'last_both':
                encoder_state_ = tf.concat([last_forward, last_backward], axis=1)
            elif encoder.final_state == 'none':
                encoder_state_ = tf.zeros(shape=[batch_size, 0])
            elif encoder.bidir and not encoder.final_state == 'last_forward':  # last backward hidden state
                encoder_state_ = last_backward
            else:  # last forward hidden state
                encoder_state_ = last_forward

            if encoder.bidir and encoder.bidir_projection:
                encoder_outputs_ = dense(encoder_outputs_, cell_output_size, use_bias=False, name='bidir_projection')

            if encoder.attend_inputs:
                encoder_outputs.append(encoder_inputs_)
            elif encoder.attend_both:
                encoder_outputs.append(tf.concat([encoder_inputs_, encoder_outputs_], axis=2))
            else:
                encoder_outputs.append(encoder_outputs_)

            encoder_states.append(encoder_state_)
            new_encoder_input_length.append(encoder_input_length_)

    encoder_state = tf.concat(encoder_states, 1)
    return encoder_outputs, encoder_state, new_encoder_input_length


def compute_energy(hidden, state, encoder, time=None, input_length=None, prev_weights=None, **kwargs):
    batch_size = tf.shape(hidden)[0]
    time_steps = tf.shape(hidden)[1]

    if encoder.attn_keep_prob is not None:
        state_noise_shape = [1, tf.shape(state)[1]] if encoder.pervasive_dropout else None
        state = tf.nn.dropout(state, keep_prob=encoder.attn_keep_prob, noise_shape=state_noise_shape)
        hidden_noise_shape = [1, 1, tf.shape(hidden)[2]] if encoder.pervasive_dropout else None
        hidden = tf.nn.dropout(hidden, keep_prob=encoder.attn_keep_prob, noise_shape=hidden_noise_shape)

    if encoder.mult_attn:
        state = dense(state, encoder.attn_size, use_bias=False, name='state')
        hidden = dense(hidden, encoder.attn_size, use_bias=False, name='hidden')
        return tf.einsum('ijk,ik->ij', hidden, state)

    y = dense(state, encoder.attn_size, use_bias=not encoder.layer_norm, name='W_a')
    y = tf.expand_dims(y, axis=1)

    if encoder.layer_norm:
        y = tf.contrib.layers.layer_norm(y, scope='layer_norm_state')
        hidden = tf.contrib.layers.layer_norm(hidden, center=False, scope='layer_norm_hidden')

    y += dense(hidden, encoder.attn_size, use_bias=False, name='U_a')

    if encoder.position_bias and input_length is not None and time is not None:
        src_pos = tf.tile(tf.expand_dims(tf.range(time_steps), axis=0), [batch_size, 1])
        trg_pos = tf.tile(tf.reshape(time, [1, 1]), [batch_size, time_steps])
        src_len = tf.tile(tf.expand_dims(input_length, axis=1), [1, time_steps])  # - 1
        pos_feats = tf.to_float(tf.stack([src_pos, trg_pos, src_len], axis=2))
        pos_feats = tf.log(1 + pos_feats)

        y += dense(pos_feats, encoder.attn_size, use_bias=False, name='P_a')

    if encoder.attn_filters:
        filter_shape = [encoder.attn_filter_length * 2 + 1, 1, 1, encoder.attn_filters]
        filter_ = get_variable('filter', filter_shape)
        prev_weights = tf.reshape(prev_weights, tf.stack([batch_size, time_steps, 1, 1]))
        conv = tf.nn.conv2d(prev_weights, filter_, [1, 1, 1, 1], 'SAME')
        conv = tf.squeeze(conv, axis=2)

        y += dense(conv, encoder.attn_size, use_bias=False, name='C_a')

    v = get_variable('v_a', [encoder.attn_size])
    return tf.reduce_sum(v * tf.tanh(y), axis=2)


def global_attention(state, hidden_states, encoder, encoder_input_length, scope=None, context=None, **kwargs):
    with tf.variable_scope(scope or 'attention_{}'.format(encoder.name)):
        if context is not None and encoder.use_context:
            state = tf.concat([state, context], axis=1)

        e = compute_energy(hidden_states, state, encoder, input_length=encoder_input_length, **kwargs)
        mask = tf.sequence_mask(encoder_input_length, maxlen=tf.shape(hidden_states)[1], dtype=tf.float32)
        e *= mask

        if encoder.attn_norm_fun == 'none':
            weights = e
        elif encoder.attn_norm_fun == 'sigmoid':
            weights = tf.nn.sigmoid(e)
        elif encoder.attn_norm_fun == 'max':
            weights = tf.one_hot(tf.argmax(e, -1), depth=tf.shape(e)[1])
        else:
            e -= tf.reduce_max(e, axis=1, keep_dims=True)
            T = encoder.attn_temperature or 1.0
            exp = tf.exp(e / T) * mask
            weights = exp / tf.reduce_sum(exp, axis=-1, keep_dims=True)

        weighted_average = tf.reduce_sum(tf.expand_dims(weights, 2) * hidden_states, axis=1)

        return weighted_average, weights


def no_attention(state, hidden_states, *args, **kwargs):
    batch_size = tf.shape(state)[0]
    weighted_average = tf.zeros(shape=tf.stack([batch_size, 0]))
    weights = tf.zeros(shape=[batch_size, tf.shape(hidden_states)[1]])
    return weighted_average, weights


def average_attention(hidden_states, encoder_input_length, *args, **kwargs):
    # attention with fixed weights (average of all hidden states)
    lengths = tf.to_float(tf.expand_dims(encoder_input_length, axis=1))
    mask = tf.sequence_mask(encoder_input_length, maxlen=tf.shape(hidden_states)[1])
    weights = tf.to_float(mask) / lengths
    weighted_average = tf.reduce_sum(hidden_states * tf.expand_dims(weights, axis=2), axis=1)
    return weighted_average, weights


def last_state_attention(hidden_states, encoder_input_length, *args, **kwargs):
    weights = tf.one_hot(encoder_input_length - 1, tf.shape(hidden_states)[1])
    weights = tf.to_float(weights)

    weighted_average = tf.reduce_sum(hidden_states * tf.expand_dims(weights, axis=2), axis=1)
    return weighted_average, weights


def local_attention(state, hidden_states, encoder, encoder_input_length, pos=None, scope=None, context=None, **kwargs):
    batch_size = tf.shape(state)[0]
    attn_length = tf.shape(hidden_states)[1]

    if context is not None and encoder.use_context:
        state = tf.concat([state, context], axis=1)

    state_size = state.get_shape()[1].value

    with tf.variable_scope(scope or 'attention_{}'.format(encoder.name)):
        encoder_input_length = tf.to_float(tf.expand_dims(encoder_input_length, axis=1))

        if pos is not None:
            pos = tf.reshape(pos, [-1, 1])
            pos = tf.minimum(pos, encoder_input_length - 1)

        if pos is not None and encoder.attn_window_size > 0:
            # `pred_edits` scenario, where we know the aligned pos
            # when the windows size is non-zero, we concatenate consecutive encoder states
            # and map it to the right attention vector size.
            weights = tf.to_float(tf.one_hot(tf.to_int32(tf.squeeze(pos, axis=1)), depth=attn_length))

            weighted_average = []
            for offset in range(-encoder.attn_window_size, encoder.attn_window_size + 1):
                pos_ = pos + offset
                pos_ = tf.minimum(pos_, encoder_input_length - 1)
                pos_ = tf.maximum(pos_, 0)  # TODO: when pos is < 0, use <S> or </S>
                weights_ = tf.to_float(tf.one_hot(tf.to_int32(tf.squeeze(pos_, axis=1)), depth=attn_length))
                weighted_average_ = tf.reduce_sum(tf.expand_dims(weights_, axis=2) * hidden_states, axis=1)
                weighted_average.append(weighted_average_)

            weighted_average = tf.concat(weighted_average, axis=1)
            weighted_average = dense(weighted_average, encoder.attn_size)
        elif pos is not None:
            weights = tf.to_float(tf.one_hot(tf.to_int32(tf.squeeze(pos, axis=1)), depth=attn_length))
            weighted_average = tf.reduce_sum(tf.expand_dims(weights, axis=2) * hidden_states, axis=1)
        else:
            # Local attention of Luong et al. (http://arxiv.org/abs/1508.04025)
            wp = get_variable('Wp', [state_size, state_size])
            vp = get_variable('vp', [state_size, 1])

            pos = tf.nn.sigmoid(tf.matmul(tf.nn.tanh(tf.matmul(state, wp)), vp))
            pos = tf.floor(encoder_input_length * pos)
            pos = tf.reshape(pos, [-1, 1])
            pos = tf.minimum(pos, encoder_input_length - 1)

            idx = tf.tile(tf.to_float(tf.range(attn_length)), tf.stack([batch_size]))
            idx = tf.reshape(idx, [-1, attn_length])

            low = pos - encoder.attn_window_size
            high = pos + encoder.attn_window_size

            mlow = tf.to_float(idx < low)
            mhigh = tf.to_float(idx > high)
            m = mlow + mhigh
            m += tf.to_float(idx >= encoder_input_length)

            mask = tf.to_float(tf.equal(m, 0.0))

            e = compute_energy(hidden_states, state, encoder, input_length=encoder_input_length, **kwargs)
            weights = softmax(e, mask=mask)

            if encoder.attn_window_size > 0:
                sigma = encoder.attn_window_size / 2
                numerator = -tf.pow((idx - pos), tf.convert_to_tensor(2, dtype=tf.float32))
                div = tf.truediv(numerator, 2 * sigma ** 2)

                weights *= tf.exp(div)  # result of the truncated normal distribution
                # normalize to keep a probability distribution
                # weights /= (tf.reduce_sum(weights, axis=1, keep_dims=True) + 10e-12)

            weighted_average = tf.reduce_sum(tf.expand_dims(weights, axis=2) * hidden_states, axis=1)

        return weighted_average, weights


def attention(encoder, scope=None, **kwargs):
    attention_functions = {
        'global': global_attention,
        'local': local_attention,
        'none': no_attention,
        'average': average_attention,
        'last_state': last_state_attention
    }
    attention_function = attention_functions.get(encoder.attention_type, global_attention)

    context_vectors = []
    weights = []

    attn_heads = encoder.attn_heads or 1
    scope = scope or 'attention_{}'.format(encoder.name)
    for i in range(attn_heads):
        scope_ = scope if i == 0 else scope + '_{}'.format(i + 1)

        context_vector, weights_ = attention_function(encoder=encoder, scope=scope_, **kwargs)
        context_vectors.append(context_vector)
        weights.append(weights_)

    context_vector = tf.concat(context_vectors, axis=-1)
    weights = sum(weights) / len(weights)

    if encoder.attn_mapping:
        with tf.variable_scope(scope):
            context_vector = dense(context_vector, encoder.attn_mapping, use_bias=False, name='output')

    return context_vector, weights


def multi_attention(state, hidden_states, encoders, encoder_input_length, pos=None, aggregation_method='sum',
                    prev_weights=None, **kwargs):
    attns = []
    weights = []

    context_vector = None
    for i, (hidden, encoder, input_length) in enumerate(zip(hidden_states, encoders, encoder_input_length)):
        pos_ = pos[i] if pos is not None else None
        prev_weights_ = prev_weights[i] if prev_weights is not None else None

        hidden = beam_search.resize_like(hidden, state)
        input_length = beam_search.resize_like(input_length, state)

        context_vector, weights_ = attention(state=state, hidden_states=hidden, encoder=encoder,
                                             encoder_input_length=input_length, pos=pos_, context=context_vector,
                                             prev_weights=prev_weights_, **kwargs)
        attns.append(context_vector)
        weights.append(weights_)

    if aggregation_method == 'sum':
        context_vector = tf.reduce_sum(tf.stack(attns, axis=2), axis=2)
    else:
        context_vector = tf.concat(attns, axis=1)

    return context_vector, weights


def attention_decoder(decoder_inputs, initial_state, attention_states, encoders, decoder, encoder_input_length,
                      feed_previous=0.0, align_encoder_id=0, feed_argmax=True, training=True, **kwargs):
    """
    :param decoder_inputs: int32 tensor of shape (batch_size, output_length)
    :param initial_state: initial state of the decoder (usually the final state of the encoder),
      as a float32 tensor of shape (batch_size, initial_state_size). This state is mapped to the
      correct state size for the decoder.
    :param attention_states: list of tensors of shape (batch_size, input_length, encoder_cell_size),
      the hidden states of the encoder(s) (one tensor for each encoder).
    :param encoders: configuration of the encoders
    :param decoder: configuration of the decoder
    :param encoder_input_length: list of int32 tensors of shape (batch_size,), tells for each encoder,
     the true length of each sequence in the batch (sequences in the same batch are padded to all have the same
     length).
    :param feed_previous: scalar tensor corresponding to the probability to use previous decoder output
      instead of the ground truth as input for the decoder (1 when decoding, between 0 and 1 when training)
    :param feed_argmax: boolean tensor, when True the greedy decoder outputs the word with the highest
    probability (argmax). When False, it samples a word from the probability distribution (softmax).
    :param align_encoder_id: outputs attention weights for this encoder. Also used when predicting edit operations
    (pred_edits), to specifify which encoder reads the sequence to post-edit (MT).

    :return:
      outputs of the decoder as a tensor of shape (batch_size, output_length, decoder_cell_size)
      attention weights as a tensor of shape (output_length, encoders, batch_size, input_length)
    """

    cell_output_size, cell_state_size = get_state_size(decoder.cell_type, decoder.cell_size,
                                                       decoder.lstm_proj_size, decoder.layers)
    utils.log("{} {}".format(cell_output_size, cell_state_size))

    assert not decoder.pred_maxout_layer or cell_output_size % 2 == 0, 'cell size must be a multiple of 2'

    if decoder.use_lstm is False:
        decoder.cell_type = 'GRU'

    embedding_shape = [decoder.vocab_size, decoder.embedding_size]
    weight_scale = decoder.embedding_weight_scale or decoder.weight_scale
    if weight_scale is None:
        initializer = None  # FIXME
    elif decoder.embedding_initializer == 'uniform' or (decoder.embedding_initializer is None
                                                        and decoder.initializer == 'uniform'):
        initializer = tf.random_uniform_initializer(minval=-weight_scale, maxval=weight_scale)
    else:
        initializer = tf.random_normal_initializer(stddev=weight_scale)

    with tf.device('/cpu:0'):
        if decoder.name == "edits":
            embedding_name = "mt"
        else:
            embedding_name = decoder.name
        embedding = get_variable('embedding_{}'.format(embedding_name), shape=embedding_shape, initializer=initializer)

    input_shape = tf.shape(decoder_inputs)
    batch_size = input_shape[0]
    time_steps = input_shape[1]

    scope_name = 'decoder_{}'.format(decoder.name)
    scope_name += '/' + '_'.join(encoder.name for encoder in encoders)

    def embed(input_):
        embedded_input = tf.nn.embedding_lookup(embedding, input_)

        if decoder.use_dropout and decoder.word_keep_prob is not None:
            noise_shape = [1, 1] if decoder.pervasive_dropout else [tf.shape(input_)[0], 1]
            embedded_input = tf.nn.dropout(embedded_input, keep_prob=decoder.word_keep_prob, noise_shape=noise_shape)
        if decoder.use_dropout and decoder.embedding_keep_prob is not None:
            size = tf.shape(embedded_input)[1]
            noise_shape = [1, size] if decoder.pervasive_dropout else [tf.shape(input_)[0], size]
            embedded_input = tf.nn.dropout(embedded_input, keep_prob=decoder.embedding_keep_prob,
                                           noise_shape=noise_shape)

        return embedded_input

    def get_cell(input_size=None, reuse=False):
        cells = []

        for j in range(decoder.layers):
            input_size_ = input_size if j == 0 else cell_output_size

            if decoder.cell_type.lower() == 'lstm':
                cell = CellWrapper(BasicLSTMCell(decoder.cell_size, reuse=reuse))
            elif decoder.cell_type.lower() == 'plstm':
                cell = PLSTM(decoder.cell_size, reuse=reuse, fact_size=decoder.lstm_fact_size,
                             proj_size=decoder.lstm_proj_size)
            elif decoder.cell_type.lower() == 'dropoutgru':
                cell = DropoutGRUCell(decoder.cell_size, reuse=reuse, layer_norm=decoder.layer_norm,
                                      input_size=input_size_, input_keep_prob=decoder.rnn_input_keep_prob,
                                      state_keep_prob=decoder.rnn_state_keep_prob)
            else:
                cell = GRUCell(decoder.cell_size, reuse=reuse, layer_norm=decoder.layer_norm)

            if decoder.use_dropout and decoder.cell_type.lower() != 'dropoutgru':
                cell = DropoutWrapper(cell, input_keep_prob=decoder.rnn_input_keep_prob,
                                      output_keep_prob=decoder.rnn_output_keep_prob,
                                      state_keep_prob=decoder.rnn_state_keep_prob,
                                      variational_recurrent=decoder.pervasive_dropout,
                                      dtype=tf.float32, input_size=input_size_)
            cells.append(cell)

        if len(cells) == 1:
            return cells[0]
        else:
            return CellWrapper(MultiRNNCell(cells))

    def look(time, state, input_, prev_weights=None, pos=None, context=None):
        prev_weights_ = [prev_weights if i == align_encoder_id else None for i in range(len(encoders))]
        pos_ = None
        if decoder.pred_edits:
            pos_ = [pos if i == align_encoder_id else None for i in range(len(encoders))]
        if decoder.attn_prev_word:
            state = tf.concat([state, input_], axis=1)

        if decoder.attn_prev_attn and context is not None:
            state = tf.concat([state, context], axis=1)

        if decoder.hidden_state_scaling:
            attention_states_ = [states * decoder.hidden_state_scaling for states in attention_states]
        else:
            attention_states_ = attention_states

        parameters = dict(hidden_states=attention_states_, encoder_input_length=encoder_input_length,
                          encoders=encoders, aggregation_method=decoder.aggregation_method)
        context, new_weights = multi_attention(state, time=time, pos=pos_, prev_weights=prev_weights_, **parameters)

        if decoder.context_mapping:
            with tf.variable_scope(scope_name):
                activation = tf.nn.tanh if decoder.context_mapping_activation == 'tanh' else None
                use_bias = not decoder.context_mapping_no_bias
                context = dense(context, decoder.context_mapping, use_bias=use_bias, activation=activation,
                                name='context_mapping')

        return context, new_weights[align_encoder_id]

    def update(state, input_, context=None, symbol=None):
        if context is not None and decoder.rnn_feed_attn:
            input_ = tf.concat([input_, context], axis=1)
        input_size = input_.get_shape()[1].value

        initializer = CellInitializer(decoder.cell_size) if decoder.orthogonal_init else None
        with tf.variable_scope(tf.get_variable_scope(), initializer=initializer):
            try:
                output, new_state = get_cell(input_size)(input_, state)
            except ValueError:  # auto_reuse doesn't work with LSTM cells
                output, new_state = get_cell(input_size, reuse=True)(input_, state)

        if decoder.skip_update and decoder.pred_edits and symbol is not None:
            is_del = tf.equal(symbol, utils.DEL_ID)
            new_state = tf.where(is_del, state, new_state)

        if decoder.cell_type.lower() == 'lstm' and decoder.use_lstm_full_state:
            output = new_state

        return output, new_state

    def update_pos(pos, symbol, max_pos=None):
        if not decoder.pred_edits:
            return pos

        is_keep = tf.equal(symbol, utils.KEEP_ID)
        is_del = tf.equal(symbol, utils.DEL_ID)
        is_not_ins = tf.logical_or(is_keep, is_del)

        pos = beam_search.resize_like(pos, symbol)
        max_pos = beam_search.resize_like(max_pos, symbol)

        pos += tf.to_float(is_not_ins)
        if max_pos is not None:
            pos = tf.minimum(pos, tf.to_float(max_pos))
        return pos

    def generate(state, input_, context):
        if decoder.pred_use_lstm_state is False:  # for back-compatibility
            state = state[:, -cell_output_size:]

        projection_input = [state, context]
        if decoder.use_previous_word:
            projection_input.insert(1, input_)  # for back-compatibility

        output_ = tf.concat(projection_input, axis=1)

        if decoder.pred_deep_layer:
            deep_layer_size = decoder.pred_deep_layer_size or decoder.embedding_size
            if decoder.layer_norm:
                output_ = dense(output_, deep_layer_size, use_bias=False, name='deep_output')
                output_ = tf.contrib.layers.layer_norm(output_, activation_fn=tf.nn.tanh, scope='output_layer_norm')
            else:
                output_ = dense(output_, deep_layer_size, activation=tf.tanh, use_bias=True, name='deep_output')

            if decoder.use_dropout:
                size = tf.shape(output_)[1]
                noise_shape = [1, size] if decoder.pervasive_dropout else None
                output_ = tf.nn.dropout(output_, keep_prob=decoder.deep_layer_keep_prob, noise_shape=noise_shape)
        else:
            if decoder.pred_maxout_layer:
                maxout_size = decoder.maxout_size or cell_output_size
                output_ = dense(output_, maxout_size, use_bias=True, name='maxout')
                if decoder.old_maxout:  # for back-compatibility with old models
                    output_ = tf.nn.pool(tf.expand_dims(output_, axis=2), window_shape=[2], pooling_type='MAX',
                                         padding='SAME', strides=[2])
                    output_ = tf.squeeze(output_, axis=2)
                else:
                    output_ = tf.maximum(*tf.split(output_, num_or_size_splits=2, axis=1))

            if decoder.pred_embed_proj:
                # intermediate projection to embedding size (before projecting to vocabulary size)
                # this is useful to reduce the number of parameters, and
                # to use the output embeddings for output projection (tie_embeddings parameter)
                output_ = dense(output_, decoder.embedding_size, use_bias=False, name='softmax0')

        if decoder.tie_embeddings and (decoder.pred_embed_proj or decoder.pred_deep_layer):
            bias = get_variable('softmax1/bias', shape=[decoder.vocab_size])
            output_ = tf.matmul(output_, tf.transpose(embedding)) + bias
        else:
            output_ = dense(output_, decoder.vocab_size, use_bias=True, name='softmax1')
        return output_

    if decoder.use_dropout:  # FIXME: why no pervasive dropout here?
        initial_state = tf.nn.dropout(initial_state, keep_prob=decoder.initial_state_keep_prob)

    with tf.variable_scope(scope_name):
        activation_fn = None if decoder.initial_state == 'linear' else tf.nn.tanh
        if decoder.initial_state == 'trained':
            initial_state = get_variable(shape=[cell_state_size], name='initial_state')
            initial_state = tf.tile(tf.expand_dims(initial_state, axis=0), [batch_size, 1])
        elif decoder.initial_state == 'zero':
            initial_state = tf.zeros(shape=[batch_size, cell_state_size])
        elif decoder.layer_norm:
            initial_state = dense(initial_state, cell_state_size, use_bias=False, name='initial_state_projection')
            initial_state = tf.contrib.layers.layer_norm(initial_state, activation_fn=activation_fn,
                                                         scope='initial_state_layer_norm')
        else:
            initial_state = dense(initial_state, cell_state_size, use_bias=True, name='initial_state_projection',
                                  activation=activation_fn)

    if decoder.cell_type.lower() == 'lstm' and decoder.use_lstm_full_state:
        initial_output = initial_state
    else:
        # Last layer's state is the right-most part. Output is the left-most part of an LSTM's state.
        initial_output = initial_state[:, -cell_output_size:]

    time = tf.constant(0, dtype=tf.int32, name='time')
    outputs = tf.TensorArray(dtype=tf.float32, size=time_steps)
    samples = tf.TensorArray(dtype=tf.int64, size=time_steps)
    inputs = tf.TensorArray(dtype=tf.int64, size=time_steps).unstack(tf.to_int64(tf.transpose(decoder_inputs)))

    states = tf.TensorArray(dtype=tf.float32, size=time_steps)
    weights = tf.TensorArray(dtype=tf.float32, size=time_steps)
    attns = tf.TensorArray(dtype=tf.float32, size=time_steps)

    initial_symbol = inputs.read(0)  # first symbol is BOS
    initial_input = embed(initial_symbol)
    initial_pos = tf.zeros([batch_size], tf.float32)
    initial_weights = tf.zeros(tf.shape(attention_states[align_encoder_id])[:2])
    zero_context = tf.zeros(shape=tf.shape(attention_states[align_encoder_id][:, 0]))  # FIXME

    with tf.variable_scope('decoder_{}'.format(decoder.name)):
        initial_context, _ = look(0, initial_output, initial_input, pos=initial_pos, prev_weights=initial_weights,
                                  context=zero_context)
    initial_data = tf.concat([initial_state, initial_context, tf.expand_dims(initial_pos, axis=1), initial_weights],
                             axis=1)
    context_size = initial_context.shape[1].value

    def get_logits(state, ids, time):  # for beam-search decoding
        with tf.variable_scope('decoder_{}'.format(decoder.name)):
            state, context, pos, prev_weights = tf.split(state, [cell_state_size, context_size, 1, -1], axis=1)
            input_ = embed(ids)

            pos = tf.squeeze(pos, axis=1)
            pos = tf.cond(tf.equal(time, 0),
                          lambda: pos,
                          lambda: update_pos(pos, ids, encoder_input_length[align_encoder_id]))

            if decoder.cell_type.lower() == 'lstm' and decoder.use_lstm_full_state:
                output = state
            else:
                # Output is always the right-most part of the state (even with multi-layer RNNs)
                # However, this only works at test time, because different dropout operations can be used
                # on state and output.
                output = state[:, -cell_output_size:]

            if decoder.conditional_rnn:
                with tf.variable_scope('conditional_1'):
                    output, state = update(state, input_)
            elif decoder.update_first:
                output, state = update(state, input_, None, ids)
            elif decoder.generate_first:
                output, state = tf.cond(tf.equal(time, 0),
                                        lambda: (output, state),
                                        lambda: update(state, input_, context, ids))

            context, new_weights = look(time, output, input_, pos=pos, prev_weights=prev_weights, context=context)

            if decoder.conditional_rnn:
                with tf.variable_scope('conditional_2'):
                    output, state = update(state, context)
            elif not decoder.generate_first:
                output, state = update(state, input_, context, ids)

            logits = generate(output, input_, context)

            pos = tf.expand_dims(pos, axis=1)
            state = tf.concat([state, context, pos, new_weights], axis=1)
            return state, logits

    def _time_step(time, input_, input_symbol, pos, state, output, outputs, states, weights, attns, prev_weights,
                   samples, context):
        if decoder.conditional_rnn:
            with tf.variable_scope('conditional_1'):
                output, state = update(state, input_)
        elif decoder.update_first:
            output, state = update(state, input_, None, input_symbol)

        context, new_weights = look(time, output, input_, pos=pos, prev_weights=prev_weights, context=context)

        if decoder.conditional_rnn:
            with tf.variable_scope('conditional_2'):
                output, state = update(state, context)
        elif not decoder.generate_first:
            output, state = update(state, input_, context, input_symbol)

        output_ = generate(output, input_, context)

        argmax = lambda: tf.argmax(output_, 1)
        target = lambda: inputs.read(time + 1)
        softmax = lambda: tf.squeeze(tf.multinomial(tf.log(tf.nn.softmax(output_)), num_samples=1),
                                     axis=1)

        use_target = tf.logical_and(time < time_steps - 1, tf.random_uniform([]) >= feed_previous)
        predicted_symbol = tf.case([
            (use_target, target),
            (tf.logical_not(feed_argmax), softmax)],
            default=argmax)  # default case is useful for beam-search

        predicted_symbol.set_shape([None])
        predicted_symbol = tf.stop_gradient(predicted_symbol)

        input_ = embed(predicted_symbol)
        pos = update_pos(pos, predicted_symbol, encoder_input_length[align_encoder_id])

        samples = samples.write(time, predicted_symbol)
        attns = attns.write(time, context)
        weights = weights.write(time, new_weights)
        states = states.write(time, state)
        outputs = outputs.write(time, output_)

        if not decoder.conditional_rnn and not decoder.update_first and decoder.generate_first:
            output, state = update(state, input_, context, predicted_symbol)

        return (time + 1, input_, predicted_symbol, pos, state, output, outputs, states, weights, attns, new_weights,
                samples, context)

    with tf.variable_scope('decoder_{}'.format(decoder.name)):
        _, _, _, new_pos, new_state, _, outputs, states, weights, attns, new_weights, samples, _ = tf.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_time_step,
            loop_vars=(time, initial_input, initial_symbol, initial_pos, initial_state, initial_output, outputs,
                       weights, states, attns, initial_weights, samples, initial_context),
            parallel_iterations=decoder.parallel_iterations,
            swap_memory=decoder.swap_memory)

    outputs = outputs.stack()
    weights = weights.stack()  # batch_size, encoders, output time, input time
    states = states.stack()
    attns = attns.stack()
    samples = samples.stack()

    # put batch_size as first dimension
    outputs = tf.transpose(outputs, perm=(1, 0, 2))
    weights = tf.transpose(weights, perm=(1, 0, 2))
    states = tf.transpose(states, perm=(1, 0, 2))
    attns = tf.transpose(attns, perm=(1, 0, 2))
    samples = tf.transpose(samples)

    return outputs, weights, states, attns, samples, get_logits, initial_data


def attention_execution_decoder(decoder_inputs, initial_state, attention_states, encoders, decoder,
                                encoder_input_length,
                                feed_previous=0.0, align_encoder_id=0, feed_argmax=True, training=True, **kwargs):
    """
    :param decoder_inputs: int32 tensor of shape (batch_size, output_length)
    :param initial_state: initial state of the decoder (usually the final state of the encoder),
      as a float32 tensor of shape (batch_size, initial_state_size). This state is mapped to the
      correct state size for the decoder.
    :param attention_states: list of tensors of shape (batch_size, input_length, encoder_cell_size),
      the hidden states of the encoder(s) (one tensor for each encoder).
    :param encoders: configuration of the encoders
    :param decoder: configuration of the decoder
    :param encoder_input_length: list of int32 tensors of shape (batch_size,), tells for each encoder,
     the true length of each sequence in the batch (sequences in the same batch are padded to all have the same
     length).
    :param feed_previous: scalar tensor corresponding to the probability to use previous decoder output
      instead of the ground truth as input for the decoder (1 when decoding, between 0 and 1 when training)
    :param feed_argmax: boolean tensor, when True the greedy decoder outputs the word with the highest
    probability (argmax). When False, it samples a word from the probability distribution (softmax).
    :param align_encoder_id: outputs attention weights for this encoder. Also used when predicting edit operations
    (pred_edits), to specifify which encoder reads the sequence to post-edit (MT).

    :return:
      outputs of the decoder as a tensor of shape (batch_size, output_length, decoder_cell_size)
      attention weights as a tensor of shape (output_length, encoders, batch_size, input_length)
    """

    cell_output_size, cell_state_size = get_state_size(decoder.cell_type, decoder.cell_size,
                                                       decoder.lstm_proj_size, decoder.layers)

    assert not decoder.pred_maxout_layer or cell_output_size % 2 == 0, 'cell size must be a multiple of 2'

    if decoder.use_lstm is False:
        decoder.cell_type = 'GRU'

    embedding_shape = [decoder.vocab_size, decoder.embedding_size]
    weight_scale = decoder.embedding_weight_scale or decoder.weight_scale
    if weight_scale is None:
        initializer = None  # FIXME
    elif decoder.embedding_initializer == 'uniform' or (decoder.embedding_initializer is None
                                                        and decoder.initializer == 'uniform'):
        initializer = tf.random_uniform_initializer(minval=-weight_scale, maxval=weight_scale)
    else:
        initializer = tf.random_normal_initializer(stddev=weight_scale)

    with tf.device('/cpu:0'):
        if decoder.name == "edits":
            embedding_name = "mt"
        else:
            embedding_name = decoder.name
        embedding = get_variable('embedding_{}'.format(embedding_name), shape=embedding_shape, initializer=initializer)

    input_shape = tf.shape(decoder_inputs)
    batch_size = input_shape[0]
    time_steps = input_shape[1]

    scope_name = 'decoder_{}'.format(decoder.name)
    scope_name += '/' + '_'.join(encoder.name for encoder in encoders)

    def embed(input_):
        embedded_input = tf.nn.embedding_lookup(embedding, input_)

        if decoder.use_dropout and decoder.word_keep_prob is not None:
            noise_shape = [1, 1] if decoder.pervasive_dropout else [tf.shape(input_)[0], 1]
            embedded_input = tf.nn.dropout(embedded_input, keep_prob=decoder.word_keep_prob, noise_shape=noise_shape)
        if decoder.use_dropout and decoder.embedding_keep_prob is not None:
            size = tf.shape(embedded_input)[1]
            noise_shape = [1, size] if decoder.pervasive_dropout else [tf.shape(input_)[0], size]
            embedded_input = tf.nn.dropout(embedded_input, keep_prob=decoder.embedding_keep_prob,
                                           noise_shape=noise_shape)

        return embedded_input

    def get_cell(input_size=None, reuse=False):
        cells = []

        for j in range(decoder.layers):
            input_size_ = input_size if j == 0 else cell_output_size

            if decoder.cell_type.lower() == 'lstm':
                cell = CellWrapper(BasicLSTMCell(decoder.cell_size, reuse=reuse))
            elif decoder.cell_type.lower() == 'plstm':
                cell = PLSTM(decoder.cell_size, reuse=reuse, fact_size=decoder.lstm_fact_size,
                             proj_size=decoder.lstm_proj_size)
            elif decoder.cell_type.lower() == 'dropoutgru':
                cell = DropoutGRUCell(decoder.cell_size, reuse=reuse, layer_norm=decoder.layer_norm,
                                      input_size=input_size_, input_keep_prob=decoder.rnn_input_keep_prob,
                                      state_keep_prob=decoder.rnn_state_keep_prob)
            else:
                cell = GRUCell(decoder.cell_size, reuse=reuse, layer_norm=decoder.layer_norm)

            if decoder.use_dropout and decoder.cell_type.lower() != 'dropoutgru':
                cell = DropoutWrapper(cell, input_keep_prob=decoder.rnn_input_keep_prob,
                                      output_keep_prob=decoder.rnn_output_keep_prob,
                                      state_keep_prob=decoder.rnn_state_keep_prob,
                                      variational_recurrent=decoder.pervasive_dropout,
                                      dtype=tf.float32, input_size=input_size_)
            cells.append(cell)

        if len(cells) == 1:
            return cells[0]
        else:
            return CellWrapper(MultiRNNCell(cells))

    def look(time, state, input_, prev_weights=None, pos=None, context=None):
        prev_weights_ = [prev_weights if i == align_encoder_id else None for i in range(len(encoders))]
        pos_ = None
        if decoder.pred_edits:
            pos_ = [pos if i == align_encoder_id else None for i in range(len(encoders))]
        if decoder.attn_prev_word:
            state = tf.concat([state, input_], axis=1)

        if decoder.attn_prev_attn and context is not None:
            state = tf.concat([state, context], axis=1)

        # attention_states = encoder hidden states
        if decoder.hidden_state_scaling:
            attention_states_ = [states * decoder.hidden_state_scaling for states in attention_states]
        else:
            attention_states_ = attention_states

        parameters = dict(hidden_states=attention_states_, encoder_input_length=encoder_input_length,
                          encoders=encoders, aggregation_method=decoder.aggregation_method)
        context, new_weights = multi_attention(state, time=time, pos=pos_, prev_weights=prev_weights_, **parameters)

        if decoder.context_mapping:
            with tf.variable_scope(scope_name):
                activation = tf.nn.tanh if decoder.context_mapping_activation == 'tanh' else None
                use_bias = not decoder.context_mapping_no_bias
                context = dense(context, decoder.context_mapping, use_bias=use_bias, activation=activation,
                                name='context_mapping')

        return context, new_weights[align_encoder_id]

    def update(state, input_, context=None, symbol=None, lm_state=None):
        # TODO: add g_i (last hidden state of RNN LM)
        if context is not None and lm_state is not None and decoder.rnn_feed_attn:
            input_ = tf.concat([input_, context, lm_state], axis=1)
        input_size = input_.get_shape()[1].value

        initializer = CellInitializer(decoder.cell_size) if decoder.orthogonal_init else None
        with tf.variable_scope(tf.get_variable_scope(), initializer=initializer):
            try:
                output, new_state = get_cell(input_size)(input_, state)
            except ValueError:  # auto_reuse doesn't work with LSTM cells
                output, new_state = get_cell(input_size, reuse=True)(input_, state)

        if decoder.skip_update and decoder.pred_edits and symbol is not None:
            is_del = tf.equal(symbol, utils.DEL_ID)
            new_state = tf.where(is_del, state, new_state)

        if decoder.cell_type.lower() == 'lstm' and decoder.use_lstm_full_state:
            output = new_state

        return output, new_state

    def update_pos(pos, symbol, max_pos=None):
        if not decoder.pred_edits:
            return pos

        is_keep = tf.equal(symbol, utils.KEEP_ID)
        is_del = tf.equal(symbol, utils.DEL_ID)
        is_not_ins = tf.logical_or(is_keep, is_del)

        pos = beam_search.resize_like(pos, symbol)
        max_pos = beam_search.resize_like(max_pos, symbol)

        pos += tf.to_float(is_not_ins)
        if max_pos is not None:
            pos = tf.minimum(pos, tf.to_float(max_pos))
        return pos

    def generate(state, input_, context):
        if decoder.pred_use_lstm_state is False:  # for back-compatibility
            state = state[:, -cell_output_size:]

        projection_input = [state, context]
        if decoder.use_previous_word:
            projection_input.insert(1, input_)  # for back-compatibility

        output_ = tf.concat(projection_input, axis=1)

        if decoder.pred_deep_layer:
            deep_layer_size = decoder.pred_deep_layer_size or decoder.embedding_size
            if decoder.layer_norm:
                output_ = dense(output_, deep_layer_size, use_bias=False, name='deep_output')
                output_ = tf.contrib.layers.layer_norm(output_, activation_fn=tf.nn.tanh, scope='output_layer_norm')
            else:
                output_ = dense(output_, deep_layer_size, activation=tf.tanh, use_bias=True, name='deep_output')

            if decoder.use_dropout:
                size = tf.shape(output_)[1]
                noise_shape = [1, size] if decoder.pervasive_dropout else None
                output_ = tf.nn.dropout(output_, keep_prob=decoder.deep_layer_keep_prob, noise_shape=noise_shape)
        else:
            if decoder.pred_maxout_layer:
                maxout_size = decoder.maxout_size or cell_output_size
                output_ = dense(output_, maxout_size, use_bias=True, name='maxout')
                if decoder.old_maxout:  # for back-compatibility with old models
                    output_ = tf.nn.pool(tf.expand_dims(output_, axis=2), window_shape=[2], pooling_type='MAX',
                                         padding='SAME', strides=[2])
                    output_ = tf.squeeze(output_, axis=2)
                else:
                    output_ = tf.maximum(*tf.split(output_, num_or_size_splits=2, axis=1))

            if decoder.pred_embed_proj:
                # intermediate projection to embedding size (before projecting to vocabulary size)
                # this is useful to reduce the number of parameters, and
                # to use the output embeddings for output projection (tie_embeddings parameter)
                output_ = dense(output_, decoder.embedding_size, use_bias=False, name='softmax0')

        if decoder.tie_embeddings and (decoder.pred_embed_proj or decoder.pred_deep_layer):
            bias = get_variable('softmax1/bias', shape=[decoder.vocab_size])
            output_ = tf.matmul(output_, tf.transpose(embedding)) + bias
        else:
            output_ = dense(output_, decoder.vocab_size, use_bias=True, name='softmax1')
        return output_

    def execute(symbol, input, lm_state):
        # predicted_symbol = KEEP -> feed input to RNN_LM
        # predicted_symbol = DEL -> do nothing, return current lm_state
        # predicted_symbol = new word -> feed that word to RNN_LM
        is_keep = tf.equal(symbol, utils.KEEP_ID)
        is_del = tf.equal(symbol, utils.DEL_ID)
        is_not_ins = tf.logical_or(is_keep, is_del)

        new_input = tf.where(is_not_ins, embed(input), embed(symbol))
        input_size = new_input.get_shape()[1].value
        initializer = CellInitializer(decoder.cell_size) if decoder.orthogonal_init else None
        with tf.variable_scope(tf.get_variable_scope(), initializer=initializer):
            try:
                lm_output, new_lm_state = get_cell(input_size)(new_input, lm_state)
            except ValueError:  # auto_reuse doesn't work with LSTM cells
                lm_output, new_lm_state = get_cell(input_size, reuse=True)(new_input, lm_state)
        if decoder.skip_update and decoder.pred_edits and symbol is not None:
            new_lm_state = tf.where(is_del, lm_state, new_lm_state)

        return lm_output, new_lm_state

    if decoder.use_dropout:  # FIXME: why no pervasive dropout here?
        initial_state = tf.nn.dropout(initial_state, keep_prob=decoder.initial_state_keep_prob)

    with tf.variable_scope(scope_name):
        activation_fn = None if decoder.initial_state == 'linear' else tf.nn.tanh
        if decoder.initial_state == 'trained':
            initial_state = get_variable(shape=[cell_state_size], name='initial_state')
            initial_state = tf.tile(tf.expand_dims(initial_state, axis=0), [batch_size, 1])
        elif decoder.initial_state == 'zero':
            initial_state = tf.zeros(shape=[batch_size, cell_state_size])
        elif decoder.layer_norm:
            initial_state = dense(initial_state, cell_state_size, use_bias=False, name='initial_state_projection')
            initial_state = tf.contrib.layers.layer_norm(initial_state, activation_fn=activation_fn,
                                                         scope='initial_state_layer_norm')
        else:
            initial_state = dense(initial_state, cell_state_size, use_bias=True, name='initial_state_projection',
                                  activation=activation_fn)

    if decoder.cell_type.lower() == 'lstm' and decoder.use_lstm_full_state:
        initial_output = initial_state
    else:
        # Last layer's state is the right-most part. Output is the left-most part of an LSTM's state.
        initial_output = initial_state[:, -cell_output_size:]

    time = tf.constant(0, dtype=tf.int32, name='time')
    outputs = tf.TensorArray(dtype=tf.float32, size=time_steps)
    samples = tf.TensorArray(dtype=tf.int64, size=time_steps)
    inputs = tf.TensorArray(dtype=tf.int64, size=time_steps).unstack(tf.to_int64(tf.transpose(decoder_inputs)))

    states = tf.TensorArray(dtype=tf.float32, size=time_steps)
    weights = tf.TensorArray(dtype=tf.float32, size=time_steps)
    attns = tf.TensorArray(dtype=tf.float32, size=time_steps)

    encoder_inputs = kwargs['encoder_inputs'][0]
    mt_inputs = decoder_inputs
    max_pos = time_steps
    # mt_inputs = tf.transpose(encoder_inputs, [1, 0])  # length x batch size
    index_range = tf.range(batch_size)
    initial_lm_state = tf.zeros(shape=[batch_size, cell_state_size])

    initial_symbol = inputs.read(0)  # first symbol is BOS
    initial_input = embed(initial_symbol)
    initial_pos = tf.zeros([batch_size], tf.float32)
    initial_weights = tf.zeros(tf.shape(attention_states[align_encoder_id])[:2])
    zero_context = tf.zeros(shape=tf.shape(attention_states[align_encoder_id][:, 0]))  # FIXME

    with tf.variable_scope('decoder_{}'.format(decoder.name)):
        initial_context, _ = look(0, initial_output, initial_input, pos=initial_pos, prev_weights=initial_weights,
                                  context=zero_context)
    # TODO: initial_data [c_i, c'_i, g_i]
    initial_data = tf.concat([initial_state, initial_context, initial_lm_state, tf.expand_dims(initial_pos, axis=1), initial_weights],
                             axis=1)
    context_size = initial_context.shape[1].value
    lm_state_size = initial_lm_state.shape[1].value

    def get_logits(state, ids, time):  # for beam-search decoding
        with tf.variable_scope('decoder_{}'.format(decoder.name)):
            state, context, lm_state, pos, prev_weights = tf.split(state, [cell_state_size, context_size, lm_state_size, 1, -1], axis=1)
            input_ = embed(ids)

            pos = tf.squeeze(pos, axis=1)
            pos = tf.cond(tf.equal(time, 0),
                          lambda: pos,
                          lambda: update_pos(pos, ids, encoder_input_length[align_encoder_id]))

            if decoder.cell_type.lower() == 'lstm' and decoder.use_lstm_full_state:
                output = state
            else:
                # Output is always the right-most part of the state (even with multi-layer RNNs)
                # However, this only works at test time, because different dropout operations can be used
                # on state and output.
                output = state[:, -cell_output_size:]

            if decoder.conditional_rnn:
                with tf.variable_scope('conditional_1'):
                    output, state = update(state, input_)
            elif decoder.update_first:
                output, state = update(state, input_, None, ids, lm_state)
            elif decoder.generate_first:
                output, state = tf.cond(tf.equal(time, 0),
                                        lambda: (output, state),
                                        lambda: update(state, input_, context, ids, lm_state))

            context, new_weights = look(time, output, input_, pos=pos, prev_weights=prev_weights, context=context)

            if decoder.conditional_rnn:
                with tf.variable_scope('conditional_2'):
                    output, state = update(state, context)
            elif not decoder.generate_first:
                output, state = update(state, input_, context, ids)

            logits = generate(output, input_, context)

            argmax = lambda: tf.argmax(logits, 1)
            target = lambda: inputs.read(time + 1)
            softmax = lambda: tf.squeeze(tf.multinomial(tf.log(tf.nn.softmax(logits)), num_samples=1),
                                         axis=1)

            use_target = tf.logical_and(time < time_steps - 1, tf.random_uniform([]) >= feed_previous)
            predicted_symbol = tf.case([
                (use_target, target),
                (tf.logical_not(feed_argmax), softmax)],
                default=argmax)  # default case is useful for beam-search

            with tf.variable_scope("rnn_lm"):
                lm_output, lm_state = execute(predicted_symbol, ids, lm_state)

            pos = tf.expand_dims(pos, axis=1)
            state = tf.concat([state, context, lm_state, pos, new_weights], axis=1)
            return state, logits

    def _time_step(time, input_, input_symbol, pos, state, output, outputs, states, weights, attns, prev_weights,
                   samples, context, lm_state):
        if decoder.conditional_rnn:
            with tf.variable_scope('conditional_1'):
                output, state = update(state, input_)
        elif decoder.update_first:
            with tf.variable_scope('op_decoder_rnn'):
                output, state = update(state, input_, None, input_symbol, lm_state=lm_state)

        # compute attention and context vector: c_src
        context, new_weights = look(time, output, input_, pos=pos, prev_weights=prev_weights, context=context)

        # feed to LSTM cell
        if decoder.conditional_rnn:
            with tf.variable_scope('conditional_2'):
                output, state = update(state, context)
        elif not decoder.generate_first:
            with tf.variable_scope('op_decoder_rnn'):
                output, state = update(state, input_, context, input_symbol, lm_state=lm_state)

        # generate operation (projection output to vocab)
        output_ = generate(output, input_, context)

        argmax = lambda: tf.argmax(output_, 1)
        target = lambda: inputs.read(time + 1)
        softmax = lambda: tf.squeeze(tf.multinomial(tf.log(tf.nn.softmax(output_)), num_samples=1),
                                     axis=1)

        use_target = tf.logical_and(time < time_steps - 1, tf.random_uniform([]) >= feed_previous)
        predicted_symbol = tf.case([
            (use_target, target),
            (tf.logical_not(feed_argmax), softmax)],
            default=argmax)  # default case is useful for beam-search

        predicted_symbol.set_shape([None])
        predicted_symbol = tf.stop_gradient(predicted_symbol)

        # TODO: feed predicted_symbol to RNN LM
        index = tf.stack([index_range, tf.cast(pos, tf.int32)], axis=1)
        current_mt_symbol = tf.gather_nd(mt_inputs, index)
        with tf.variable_scope("rnn_lm"):
            lm_output, lm_state = execute(predicted_symbol, current_mt_symbol, lm_state)

        input_ = embed(predicted_symbol)
        pos = update_pos(pos, predicted_symbol, encoder_input_length[align_encoder_id])

        samples = samples.write(time, predicted_symbol)
        attns = attns.write(time, context)
        weights = weights.write(time, new_weights)
        states = states.write(time, state)
        outputs = outputs.write(time, output_)

        if not decoder.conditional_rnn and not decoder.update_first and decoder.generate_first:
            output, state = update(state, input_, context, predicted_symbol, lm_state=lm_state)

        return (time + 1, input_, predicted_symbol, pos, state, output, outputs, states, weights, attns, new_weights,
                samples, context, lm_state)

    with tf.variable_scope('decoder_{}'.format(decoder.name)):
        # TODO: check order states and weights
        _, _, _, new_pos, new_state, _, outputs, states, weights, attns, new_weights, samples, _, _ = tf.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_time_step,
            loop_vars=(time, initial_input, initial_symbol, initial_pos, initial_state, initial_output, outputs,
                       weights, states, attns, initial_weights, samples, initial_context, initial_lm_state),
            parallel_iterations=decoder.parallel_iterations,
            swap_memory=decoder.swap_memory)

    outputs = outputs.stack()
    weights = weights.stack()  # batch_size, encoders, output time, input time
    states = states.stack()
    attns = attns.stack()
    samples = samples.stack()

    # put batch_size as first dimension
    outputs = tf.transpose(outputs, perm=(1, 0, 2))
    weights = tf.transpose(weights, perm=(1, 0, 2))
    states = tf.transpose(states, perm=(1, 0, 2))
    attns = tf.transpose(attns, perm=(1, 0, 2))
    samples = tf.transpose(samples)

    return outputs, weights, states, attns, samples, get_logits, initial_data


def encoder_decoder(encoders, decoders, encoder_inputs, targets, feed_previous, align_encoder_id=0,
                    encoder_input_length=None, feed_argmax=True, rewards=None, use_baseline=True,
                    training=True, global_step=None,
                    monotonicity_weight=None, monotonicity_dist=None, monotonicity_decay=None, **kwargs):
    decoder = decoders[0]
    targets = targets[0]  # single decoder

    if encoder_input_length is None:
        encoder_input_length = []
        for encoder_inputs_ in encoder_inputs:
            mask = get_weights(encoder_inputs_, utils.EOS_ID, include_first_eos=True)
            encoder_input_length.append(tf.to_int32(tf.reduce_sum(mask, axis=1)))

    parameters = dict(encoders=encoders, decoder=decoder, encoder_inputs=encoder_inputs,
                      feed_argmax=feed_argmax, training=training)

    attention_states, encoder_state, encoder_input_length = multi_encoder(
        encoder_input_length=encoder_input_length, **parameters)

    outputs, attention_weights, _, _, samples, beam_fun, initial_data = attention_decoder(
        attention_states=attention_states, initial_state=encoder_state, feed_previous=feed_previous,
        decoder_inputs=targets[:, :-1], align_encoder_id=align_encoder_id, encoder_input_length=encoder_input_length,
        **parameters
    )

    if use_baseline:
        baseline_rewards = reinforce_baseline(outputs, rewards)  # FIXME: use logits or decoder outputs?
        baseline_weights = get_weights(samples, utils.EOS_ID, include_first_eos=False)
        baseline_loss_ = baseline_loss(rewards=baseline_rewards, weights=baseline_weights)
    else:
        baseline_rewards = rewards
        baseline_loss_ = tf.constant(0.0)

    reinforce_weights = get_weights(samples, utils.EOS_ID, include_first_eos=True)
    reinforce_loss = sequence_loss(logits=outputs, targets=samples, weights=reinforce_weights,
                                   rewards=baseline_rewards)

    trg_mask = get_weights(targets[:, 1:], utils.EOS_ID, include_first_eos=True)
    xent_loss = sequence_loss(logits=outputs, targets=targets[:, 1:], weights=trg_mask)

    if monotonicity_weight:
        monotonicity_dist = monotonicity_dist or 1.0

        batch_size = tf.shape(attention_weights)[0]
        src_len = tf.shape(attention_weights)[2]
        trg_len = tf.shape(attention_weights)[1]

        src_indices = tf.tile(tf.reshape(tf.range(src_len), shape=[1, 1, src_len]), [batch_size, trg_len, 1])
        trg_indices = tf.tile(tf.reshape(tf.range(trg_len), shape=[1, trg_len, 1]), [batch_size, 1, src_len])

        source_length = encoder_input_length[0]
        target_length = tf.to_int32(tf.reduce_sum(trg_mask, axis=1))
        true_src_len = tf.reshape(source_length, shape=[batch_size, 1, 1]) - 1
        true_trg_len = tf.reshape(target_length, shape=[batch_size, 1, 1]) - 1

        src_mask = tf.to_float(tf.sequence_mask(source_length, maxlen=src_len))
        mask = tf.matmul(tf.expand_dims(trg_mask, axis=2), tf.expand_dims(src_mask, axis=1))

        monotonous = tf.sqrt(((true_trg_len * src_indices - true_src_len * trg_indices) ** 2)
                             / (true_trg_len ** 2 + true_src_len ** 2))
        monotonous = tf.to_float(monotonous < monotonicity_dist)
        non_monotonous = (1 - monotonous) * mask
        attn_loss = tf.reduce_sum(attention_weights * tf.stop_gradient(non_monotonous)) / tf.to_float(batch_size)

        if monotonicity_decay:
            decay = tf.stop_gradient(0.5 ** (tf.to_float(global_step) / monotonicity_decay))
        else:
            decay = 1.0

        xent_loss += monotonicity_weight * decay * attn_loss

    losses = [xent_loss, reinforce_loss, baseline_loss_]

    return losses, [outputs], encoder_state, attention_states, attention_weights, samples, beam_fun, initial_data


def reconstruction_encoder_decoder(encoders, decoders, encoder_inputs, targets, feed_previous,
                                   encoder_input_length=None, training=True, reconstruction_weight=1.0,
                                   reconstruction_attn_weight=0.05, **kwargs):
    encoders = encoders[:1]

    if encoder_input_length is None:
        weights = get_weights(encoder_inputs[0], utils.EOS_ID, include_first_eos=True)
        encoder_input_length = [tf.to_int32(tf.reduce_sum(weights, axis=1))]

    attention_states, encoder_state, encoder_input_length = multi_encoder(
        encoder_input_length=encoder_input_length, encoders=encoders, encoder_inputs=encoder_inputs,
        training=training)

    outputs, attention_weights, states, _, samples, beam_fun, initial_data = attention_decoder(
        attention_states=attention_states, initial_state=encoder_state, feed_previous=feed_previous,
        decoder_inputs=targets[0][:, :-1], encoder_input_length=encoder_input_length,
        decoder=decoders[0], training=training, encoders=encoders
    )

    target_weights = get_weights(targets[0][:, 1:], utils.EOS_ID, include_first_eos=True)
    target_length = [tf.to_int32(tf.reduce_sum(target_weights, axis=1))]

    xent_loss = sequence_loss(logits=outputs, targets=targets[0][:, 1:], weights=target_weights)

    reconstructed_outputs, reconstructed_weights, _, _, _, _, _ = attention_decoder(
        attention_states=[states], initial_state=states[:, -1, :], feed_previous=feed_previous,
        decoder_inputs=targets[1][:, :-1], encoder_input_length=target_length,
        decoder=decoders[1], training=training, encoders=decoders[:1]
    )

    target_weights = get_weights(targets[1][:, 1:], utils.EOS_ID, include_first_eos=True)
    xent_loss += reconstruction_weight * sequence_loss(logits=reconstructed_outputs, targets=targets[1][:, 1:],
                                                       weights=target_weights)

    max_src_len = tf.shape(reconstructed_weights)[1]
    batch_size = tf.shape(reconstructed_weights)[0]

    attn_loss = tf.matmul(reconstructed_weights, attention_weights) - tf.eye(max_src_len)

    src_mask = tf.sequence_mask(encoder_input_length[0], maxlen=max_src_len, dtype=tf.float32)
    src_mask = tf.einsum('ij,ik->ijk', src_mask, src_mask)
    attn_loss *= tf.to_float(src_mask)  # don't take padding words into account

    attn_loss = tf.norm(attn_loss) / tf.to_float(batch_size)
    xent_loss += reconstruction_attn_weight * attn_loss

    attention_weights = [attention_weights, reconstructed_weights]
    losses = [xent_loss, None, None]
    return losses, [outputs], encoder_state, attention_states, attention_weights, samples, beam_fun, initial_data


def chained_encoder_decoder(encoders, decoders, encoder_inputs, targets, feed_previous,
                            chaining_strategy=None, align_encoder_id=0, chaining_non_linearity=False,
                            chaining_loss_ratio=1.0, chaining_stop_gradient=False, training=True, **kwargs):
    decoder = decoders[0]
    targets = targets[0]  # single decoder

    assert len(encoders) == 2

    encoder_input_length = []
    input_weights = []
    for encoder_inputs_ in encoder_inputs:
        weights = get_weights(encoder_inputs_, utils.EOS_ID, include_first_eos=True)
        input_weights.append(weights)
        encoder_input_length.append(tf.to_int32(tf.reduce_sum(weights, axis=1)))

    target_weights = get_weights(targets[:, 1:], utils.EOS_ID, include_first_eos=True)

    parameters = dict(encoders=encoders[1:], decoder=encoders[0], training=training)

    attention_states, encoder_state, encoder_input_length[1:] = multi_encoder(
        encoder_inputs[1:], encoder_input_length=encoder_input_length[1:], **parameters)

    decoder_inputs = encoder_inputs[0][:, :-1]
    batch_size = tf.shape(decoder_inputs)[0]

    pad = tf.ones(shape=tf.stack([batch_size, 1]), dtype=tf.int32) * utils.BOS_ID
    decoder_inputs = tf.concat([pad, decoder_inputs], axis=1)

    outputs, _, states, attns, _, _, _ = attention_decoder(
        attention_states=attention_states, initial_state=encoder_state, decoder_inputs=decoder_inputs,
        encoder_input_length=encoder_input_length[1:], **parameters
    )

    chaining_loss = sequence_loss(logits=outputs, targets=encoder_inputs[0], weights=input_weights[0])

    if 'lstm' in decoder.cell_type.lower():
        size = states.get_shape()[2].value
        decoder_outputs = states[:, :, size // 2:]
    else:
        decoder_outputs = states

    if chaining_strategy == 'share_states':
        other_inputs = states
    elif chaining_strategy == 'share_outputs':
        other_inputs = decoder_outputs
    else:
        other_inputs = None

    if other_inputs is not None and chaining_stop_gradient:
        other_inputs = tf.stop_gradient(other_inputs)

    parameters = dict(encoders=encoders[:1], decoder=decoder, encoder_inputs=encoder_inputs[:1],
                      other_inputs=other_inputs, training=training)

    attention_states, encoder_state, encoder_input_length[:1] = multi_encoder(
        encoder_input_length=encoder_input_length[:1], **parameters)

    if chaining_stop_gradient:
        attns = tf.stop_gradient(attns)
        states = tf.stop_gradient(states)
        decoder_outputs = tf.stop_gradient(decoder_outputs)

    if chaining_strategy == 'concat_attns':
        attention_states[0] = tf.concat([attention_states[0], attns], axis=2)
    elif chaining_strategy == 'concat_states':
        attention_states[0] = tf.concat([attention_states[0], states], axis=2)
    elif chaining_strategy == 'sum_attns':
        attention_states[0] += attns
    elif chaining_strategy in ('map_attns', 'map_states', 'map_outputs'):
        if chaining_strategy == 'map_attns':
            x = attns
        elif chaining_strategy == 'map_outputs':
            x = decoder_outputs
        else:
            x = states

        shape = [x.get_shape()[-1], attention_states[0].get_shape()[-1]]

        w = tf.get_variable("map_attns/matrix", shape=shape)
        b = tf.get_variable("map_attns/bias", shape=shape[-1:])

        x = tf.einsum('ijk,kl->ijl', x, w) + b
        if chaining_non_linearity:
            x = tf.nn.tanh(x)

        attention_states[0] += x

    outputs, attention_weights, _, _, samples, beam_fun, initial_data = attention_decoder(
        attention_states=attention_states, initial_state=encoder_state,
        feed_previous=feed_previous, decoder_inputs=targets[:, :-1],
        align_encoder_id=align_encoder_id, encoder_input_length=encoder_input_length[:1],
        **parameters
    )

    xent_loss = sequence_loss(logits=outputs, targets=targets[:, 1:],
                              weights=target_weights)

    if chaining_loss is not None and chaining_loss_ratio:
        xent_loss += chaining_loss_ratio * chaining_loss

    losses = [xent_loss, None, None]

    return losses, [outputs], encoder_state, attention_states, attention_weights, samples, beam_fun, initial_data


def chained_encoder_decoder_execute(encoders, decoders, encoder_inputs, targets, feed_previous,
                                    chaining_strategy=None, align_encoder_id=0, chaining_non_linearity=False,
                                    chaining_loss_ratio=1.0, chaining_stop_gradient=False, training=True, **kwargs):
    decoder = decoders[0]
    targets = targets[0]  # single decoder

    assert len(encoders) == 2

    encoder_input_length = []
    input_weights = []
    for encoder_inputs_ in encoder_inputs:
        weights = get_weights(encoder_inputs_, utils.EOS_ID, include_first_eos=True)
        input_weights.append(weights)
        encoder_input_length.append(tf.to_int32(tf.reduce_sum(weights, axis=1)))

    target_weights = get_weights(targets[:, 1:], utils.EOS_ID, include_first_eos=True)

    parameters = dict(encoders=encoders[1:], decoder=encoders[0], training=training)

    # src encoder
    attention_states, encoder_state, encoder_input_length[1:] = multi_encoder(
        encoder_inputs[1:], encoder_input_length=encoder_input_length[1:], **parameters)

    decoder_inputs = encoder_inputs[0][:, :-1]
    batch_size = tf.shape(decoder_inputs)[0]

    pad = tf.ones(shape=tf.stack([batch_size, 1]), dtype=tf.int32) * utils.BOS_ID
    decoder_inputs = tf.concat([pad, decoder_inputs], axis=1)

    # src -> mt decoder
    outputs, _, states, attns, _, _, _ = attention_decoder(
        attention_states=attention_states, initial_state=encoder_state, decoder_inputs=decoder_inputs,
        encoder_input_length=encoder_input_length[1:], **parameters
    )

    chaining_loss = sequence_loss(logits=outputs, targets=encoder_inputs[0], weights=input_weights[0])

    if 'lstm' in decoder.cell_type.lower():
        size = states.get_shape()[2].value
        decoder_outputs = states[:, :, size // 2:]
    else:
        decoder_outputs = states

    if chaining_strategy == 'share_states':
        other_inputs = states
    elif chaining_strategy == 'share_outputs':
        other_inputs = decoder_outputs
    else:
        other_inputs = None

    if other_inputs is not None and chaining_stop_gradient:
        other_inputs = tf.stop_gradient(other_inputs)

    parameters = dict(encoders=encoders[:1], decoder=decoder, encoder_inputs=encoder_inputs[:1],
                      other_inputs=other_inputs, training=training)

    # mt encoder
    attention_states, encoder_state, encoder_input_length[:1] = multi_encoder(
        encoder_input_length=encoder_input_length[:1], **parameters)

    if chaining_stop_gradient:
        attns = tf.stop_gradient(attns)
        states = tf.stop_gradient(states)
        decoder_outputs = tf.stop_gradient(decoder_outputs)

    if chaining_strategy == 'concat_attns':
        attention_states[0] = tf.concat([attention_states[0], attns], axis=2)
    elif chaining_strategy == 'concat_states':
        attention_states[0] = tf.concat([attention_states[0], states], axis=2)
    elif chaining_strategy == 'sum_attns':
        attention_states[0] += attns
    elif chaining_strategy in ('map_attns', 'map_states', 'map_outputs'):
        if chaining_strategy == 'map_attns':
            x = attns
        elif chaining_strategy == 'map_outputs':
            x = decoder_outputs
        else:
            x = states

        shape = [x.get_shape()[-1], attention_states[0].get_shape()[-1]]

        w = tf.get_variable("map_attns/matrix", shape=shape)
        b = tf.get_variable("map_attns/bias", shape=shape[-1:])

        x = tf.einsum('ijk,kl->ijl', x, w) + b
        if chaining_non_linearity:
            x = tf.nn.tanh(x)

        attention_states[0] += x

    outputs, attention_weights, _, _, samples, beam_fun, initial_data = attention_execution_decoder(
        attention_states=attention_states, initial_state=encoder_state,
        feed_previous=feed_previous, decoder_inputs=targets[:, :-1],
        align_encoder_id=align_encoder_id, encoder_input_length=encoder_input_length[:1],
        **parameters
    )

    xent_loss = sequence_loss(logits=outputs, targets=targets[:, 1:],
                              weights=target_weights)

    if chaining_loss is not None and chaining_loss_ratio:
        xent_loss += chaining_loss_ratio * chaining_loss

    losses = [xent_loss, None, None]

    return losses, [outputs], encoder_state, attention_states, attention_weights, samples, beam_fun, initial_data


def softmax(logits, dim=-1, mask=None):
    e = tf.exp(logits)
    if mask is not None:
        e *= mask

    return e / tf.clip_by_value(tf.reduce_sum(e, axis=dim, keep_dims=True), 10e-37, 10e+37)


def sequence_loss(logits, targets, weights, average_across_timesteps=False, average_across_batch=True, rewards=None):
    batch_size = tf.shape(targets)[0]
    time_steps = tf.shape(targets)[1]

    logits_ = tf.reshape(logits, tf.stack([time_steps * batch_size, logits.get_shape()[2].value]))
    targets_ = tf.reshape(targets, tf.stack([time_steps * batch_size]))

    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_, labels=targets_)
    crossent = tf.reshape(crossent, tf.stack([batch_size, time_steps]))

    if rewards is not None:
        crossent *= tf.stop_gradient(rewards)

    log_perp = tf.reduce_sum(crossent * weights, axis=1)

    if average_across_timesteps:
        total_size = tf.reduce_sum(weights, axis=1)
        total_size += 1e-12  # just to avoid division by 0 for all-0 weights
        log_perp /= total_size

    cost = tf.reduce_sum(log_perp)

    if average_across_batch:
        return cost / tf.to_float(batch_size)
    else:
        return cost


def reinforce_baseline(decoder_states, reward):
    """
    Center the reward by computing a baseline reward over decoder states.

    :param decoder_states: internal states of the decoder, tensor of shape (batch_size, time_steps, state_size)
    :param reward: reward for each time step, tensor of shape (batch_size, time_steps)
    :return: reward - computed baseline, tensor of shape (batch_size, time_steps)
    """
    # batch_size = tf.shape(decoder_states)[0]
    # time_steps = tf.shape(decoder_states)[1]
    # state_size = decoder_states.get_shape()[2]
    # states = tf.reshape(decoder_states, shape=tf.stack([batch_size * time_steps, state_size]))

    baseline = dense(tf.stop_gradient(decoder_states), units=1, activation=None, name='reward_baseline',
                     kernel_initializer=tf.constant_initializer(0.01))
    baseline = tf.squeeze(baseline, axis=2)

    # baseline = tf.reshape(baseline, shape=tf.stack([batch_size, time_steps]))
    return reward - baseline


def baseline_loss(rewards, weights, average_across_timesteps=False, average_across_batch=True):
    """
    :param rewards: tensor of shape (batch_size, time_steps)
    :param weights: tensor of shape (batch_size, time_steps)
    """
    batch_size = tf.shape(rewards)[0]

    cost = rewards ** 2
    cost = tf.reduce_sum(cost * weights, axis=1)

    if average_across_timesteps:
        total_size = tf.reduce_sum(weights, axis=1)
        total_size += 1e-12  # just to avoid division by 0 for all-0 weights
        cost /= total_size

    cost = tf.reduce_sum(cost)

    if average_across_batch:
        cost /= tf.to_float(batch_size)

    return cost
