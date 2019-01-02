import numpy as np
import tensorflow as tf
import re
import functools

from translate import utils
from translate import models
from translate import evaluation
from translate import beam_search
from collections import namedtuple


class Seq2SeqModel(object):
    def __init__(self, encoders, decoders, learning_rate, global_step, max_gradient_norm, use_dropout=False,
                 freeze_variables=None, feed_previous=0.0, optimizer='sgd', decode_only=False,
                 len_normalization=1.0, name=None, chained_encoders=False, pred_edits=False, baseline_step=None,
                 use_baseline=True, reverse_input=False, reconstruction_decoders=False, chained_execute=False, **kwargs):
        self.encoders = encoders
        self.decoders = decoders
        self.temperature = self.decoders[0].temperature

        self.name = name

        self.learning_rate = learning_rate
        self.global_step = global_step
        self.baseline_step = baseline_step
        self.use_baseline = use_baseline

        self.max_output_len = [decoder.max_len for decoder in decoders]
        self.max_input_len = [encoder.max_len for encoder in encoders]
        self.len_normalization = len_normalization
        self.reverse_input = reverse_input

        dropout_on = []
        dropout_off = []

        if use_dropout:
            for encoder_or_decoder in encoders + decoders:
                names = ['rnn_input', 'rnn_output', 'rnn_state', 'initial_state', 'word', 'input_layer', 'output',
                         'attn', 'deep_layer', 'inter_layer', 'embedding']

                for name in names:
                    value = encoder_or_decoder.get(name + '_dropout')
                    var_name = name + '_keep_prob'
                    if not value:
                        encoder_or_decoder[var_name] = 1.0
                        continue
                    var = tf.Variable(1 - value, trainable=False, name=var_name)
                    encoder_or_decoder[var_name] = var
                    dropout_on.append(var.assign(1.0 - value))
                    dropout_off.append(var.assign(1.0))

        self.dropout_on = tf.group(*dropout_on)
        self.dropout_off = tf.group(*dropout_off)

        self.feed_previous = tf.constant(feed_previous, dtype=tf.float32)
        self.feed_argmax = tf.constant(True, dtype=tf.bool)  # feed with argmax or sample from softmax
        self.training = tf.placeholder(dtype=tf.bool, shape=())

        self.encoder_inputs = []
        self.encoder_input_length = []
        for encoder in encoders:
            shape = [None, None, encoder.embedding_size] if encoder.binary else [None, None]
            dtype = tf.float32 if encoder.binary else tf.int32
            encoder_input = tf.placeholder(dtype=dtype, shape=shape, name='encoder_{}'.format(encoder.name))
            encoder_input_length = tf.placeholder(dtype=tf.int32, shape=[None],
                                                  name='encoder_input_length_{}'.format(encoder.name))
            self.encoder_inputs.append(encoder_input)
            self.encoder_input_length.append(encoder_input_length)

        # starts with BOS, and ends with EOS
        self.targets = tuple([
            tf.placeholder(tf.int32, shape=[None, None], name='target_{}'.format(decoder.name))
            for decoder in decoders
        ])
        self.rewards = tf.placeholder(tf.float32, shape=[None, None], name='rewards')

        if reconstruction_decoders:
            architecture = models.reconstruction_encoder_decoder
        elif chained_encoders and pred_edits:
            if chained_execute:
                architecture = models.chained_encoder_decoder_execute
            else:
                architecture = models.chained_encoder_decoder    # no REINFORCE for now
        else:
             architecture = models.encoder_decoder

        tensors = architecture(encoders, decoders, self.encoder_inputs, self.targets, self.feed_previous,
                               encoder_input_length=self.encoder_input_length, feed_argmax=self.feed_argmax,
                               rewards=self.rewards, use_baseline=use_baseline, training=self.training,
                               global_step=self.global_step, **kwargs)

        (self.losses, self.outputs, self.encoder_state, self.attention_states, self.attention_weights,
         self.samples, self.beam_fun, self.initial_data) = tensors

        self.xent_loss, self.reinforce_loss, self.baseline_loss = self.losses
        self.loss = self.xent_loss   # main loss

        optimizers = self.get_optimizers(optimizer, learning_rate)

        if not decode_only:
            get_update_ops = functools.partial(self.get_update_op, opts=optimizers,
                                               max_gradient_norm=max_gradient_norm, freeze_variables=freeze_variables)

            self.update_ops = utils.AttrDict({
                'xent': get_update_ops(self.xent_loss, global_step=self.global_step),
                'reinforce': get_update_ops(self.reinforce_loss, global_step=self.global_step),
            })

            if use_baseline:
                self.update_ops['baseline'] = get_update_ops(self.baseline_loss, global_step=self.baseline_step)

        self.models = [self]
        self.beam_outputs = tf.expand_dims(tf.argmax(self.outputs[0], axis=2), axis=1)
        self.beam_scores = tf.zeros(shape=[tf.shape(self.beam_outputs)[0], 1])
        self.beam_size = tf.placeholder(shape=(), dtype=tf.int32)

    def create_beam_op(self, models, len_normalization):
        self.len_normalization = len_normalization
        self.models = models
        beam_funs = [model.beam_fun for model in models]
        initial_data = [model.initial_data for model in models]
        beam_output = beam_search.rnn_beam_search(beam_funs, initial_data, self.max_output_len[0], self.beam_size,
                                                  len_normalization, temperature=self.temperature,
                                                  parallel_iterations=self.decoders[0].parallel_iterations,
                                                  swap_memory=self.decoders[0].swap_memory)
        self.beam_outputs, self.beam_scores = beam_output

    @staticmethod
    def get_optimizers(optimizer_name, learning_rate):
        sgd_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        if optimizer_name.lower() == 'adadelta':
            # same epsilon and rho as Bahdanau et al. 2015
            opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, epsilon=1e-06, rho=0.95)
        elif optimizer_name.lower() == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            opt = sgd_opt

        return opt, sgd_opt

    def get_update_op(self, loss, opts, global_step=None, max_gradient_norm=None, freeze_variables=None):
        if loss is None:
            return None

        freeze_variables = freeze_variables or []

        # compute gradient only for variables that are not frozen
        frozen_parameters = [var.name for var in tf.trainable_variables()
                             if any(re.match(var_, var.name) for var_ in freeze_variables)]
        params = [var for var in tf.trainable_variables() if var.name not in frozen_parameters]
        self.params = params

        gradients = tf.gradients(loss, params)

        # from translate.memory_saving_gradients import gradients as mem_save_gradients
        # gradients = mem_save_gradients(loss, params, checkpoints='speed')  # try 'memory'

        if max_gradient_norm:
            gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

        update_ops = []
        for opt in opts:
            update_ops_ = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.variable_scope('gradients' if self.name is None else 'gradients_{}'.format(self.name)):
                with tf.control_dependencies(update_ops_):  # update batch_norm's moving averages
                    update_op = opt.apply_gradients(list(zip(gradients, params)), global_step=global_step)

            update_ops.append(update_op)

        return update_ops

    def reinforce_step(self, data, update_model=True, align=False, use_sgd=False, update_baseline=True,
                       reward_function=None, **kwargs):
        # self.dropout_on.run()

        encoder_inputs, targets, input_length = self.get_batch(data)
        input_feed = {self.targets: targets, self.feed_argmax: False, self.feed_previous: 1.0, self.training: True}

        for i in range(len(self.encoders)):
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]
            input_feed[self.encoder_input_length[i]] = input_length[i]

        samples, outputs = tf.get_default_session().run([self.samples, self.outputs], input_feed)

        if reward_function is None:
            reward_function = 'sentence_bleu'
        reward_function = getattr(evaluation, reward_function)

        def compute_reward(output, target):
            j, = np.where(output == utils.EOS_ID)  # array of indices whose value is EOS_ID
            if len(j) > 0:
                output = output[:j[0]]

            j, = np.where(target == utils.EOS_ID)
            if len(j) > 0:
                target = target[:j[0]]

            return reward_function(output, target)

        def compute_rewards(outputs, targets):
            return np.array([compute_reward(output, target) for output, target in zip(outputs, targets)])

        rewards = compute_rewards(samples, targets[0][:,1:])
        rewards = np.stack([rewards] * samples.shape[1], axis=1)

        input_feed[self.outputs[0]] = outputs[0]
        input_feed[self.samples] = samples
        input_feed[self.rewards] = rewards

        output_feed = {'loss': self.reinforce_loss, 'baseline_loss': self.baseline_loss}
        if update_model:
            output_feed['update'] = self.update_ops.reinforce[1] if use_sgd else self.update_ops.reinforce[0]
        if self.use_baseline and update_baseline:
            output_feed['baseline_update'] = self.update_ops.baseline[0]  # FIXME

        if align:
            output_feed['weights'] = self.attention_weights

        res = tf.get_default_session().run(output_feed, input_feed)
        return namedtuple('output', 'loss weights baseline_loss')(res['loss'], res.get('weights'),
                                                                  res.get('baseline_loss'))

    def step(self, data, update_model=True, align=False, use_sgd=False, **kwargs):
        if update_model:
            self.dropout_on.run()
        else:
            self.dropout_off.run()

        encoder_inputs, targets, input_length = self.get_batch(data)
        input_feed = {self.targets: targets, self.training: True}

        for i in range(len(self.encoders)):
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]
            input_feed[self.encoder_input_length[i]] = input_length[i]

        output_feed = {'loss': self.xent_loss}
        if update_model:
            output_feed['update'] = self.update_ops.xent[1] if use_sgd else self.update_ops.xent[0]
        if align:
            output_feed['weights'] = self.attention_weights

        res = tf.get_default_session().run(output_feed, input_feed)
        return namedtuple('output', 'loss weights')(res['loss'], res.get('weights'))

    def greedy_decoding(self, token_ids, align=False, beam_size=1):
        for model in self.models:
            model.dropout_off.run()

        data = [
            ids + [[] for _ in self.decoders] if len(ids) == len(self.encoders) else ids
            for ids in token_ids
        ]

        batch = self.get_batch(data, decoding=True)
        encoder_inputs, targets, input_length = batch

        input_feed = {self.beam_size: beam_size}
        for model in self.models:
            input_feed[model.targets] = targets
            input_feed[model.feed_previous] = 1.0
            input_feed[model.training] = False
            for i in range(len(model.encoders)):
                input_feed[model.encoder_inputs[i]] = encoder_inputs[i]
                input_feed[model.encoder_input_length[i]] = input_length[i]

        output_feed = {'outputs': self.beam_outputs}
        if align:
            output_feed['weights'] = self.attention_weights

        res = tf.get_default_session().run(output_feed, input_feed)
        return [res['outputs'][:,0,:]], res.get('weights')

    def get_batch(self, data, decoding=False):
        """
        :param data:
        :param decoding: set this parameter to True to output dummy
          data for the decoder side (using the maximum output size)
        :return:
        """
        inputs = [[] for _ in self.encoders]
        targets = [[] for _ in self.decoders]
        input_length = [[] for _ in self.encoders]

        # maximum input length of each encoder in this batch
        max_input_len = [max(len(data_[i]) for data_ in data) for i in range(len(self.encoders))]

        if self.max_input_len is not None:
            max_input_len = [min(len_, max_len) for len_, max_len in zip(max_input_len, self.max_input_len)]

        # maximum output length in this batch
        if decoding:
            max_output_len = self.max_output_len
        else:
            max_output_len = [max(len(data_[i]) for data_ in data)
                              for i in range(len(self.encoders), len(self.encoders) + len(self.decoders))]
            if self.max_output_len is not None:
                max_output_len = [min(len_, max_len) for len_, max_len in zip(max_output_len, self.max_output_len)]

        for sentences in data:
            src_sentences = sentences[:len(self.encoders)]
            trg_sentences = sentences[len(self.encoders):]

            for i, (encoder, src_sentence) in enumerate(zip(self.encoders, src_sentences)):
                src_sentence = src_sentence[:max_input_len[i]]
                pad_symbol = np.zeros(encoder.embedding_size, dtype=np.float32) if encoder.binary else utils.EOS_ID
                # pad sequences so that all sequences in the same batch have the same length

                eos = 0 if encoder.binary else 1   # end of sentence marker for non-binary input
                encoder_pad = [pad_symbol] * (eos + max_input_len[i] - len(src_sentence))

                if self.reverse_input:
                    src_sentence = src_sentence[::-1]

                inputs[i].append(src_sentence + encoder_pad)
                input_length[i].append(len(src_sentence) + eos)

            for i in range(len(targets)):
                if decoding:
                    targets[i].append([utils.BOS_ID] * self.max_output_len[i] + [utils.EOS_ID])
                else:
                    trg_sentence = trg_sentences[i][:max_output_len[i]]
                    decoder_pad_size = max_output_len[i] - len(trg_sentence) + 1
                    trg_sentence = [utils.BOS_ID] + trg_sentence + [utils.EOS_ID] * decoder_pad_size
                    targets[i].append(trg_sentence)

        # convert lists to numpy arrays
        inputs = [np.array(inputs_, dtype=np.float32 if encoder.binary else np.int32)
                  for encoder, inputs_ in zip(self.encoders, inputs)]
        # starts with BOS and ends with EOS
        targets = [np.array(targets_, dtype=np.int32) for targets_ in targets]
        input_length = [np.array(input_length_, dtype=np.int32) for input_length_ in input_length]

        return inputs, targets, input_length
