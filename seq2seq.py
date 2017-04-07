import tensorflow as tf

from clm.model.base import (
    ClippingConstrainLanguageModel,
    prepare_rnn, linear
)


from clm.model.lang import (
    build_static_index_input,
    build_embedding_for_index,
    lm_dynamic_unfold,
    seq2seq_loss
)


def _decoder_unfold(cell, esentences, initial_state, T, is_training):
    assert len(esentences.get_shape()) == 2
    state = initial_state
    rnn_outputs = []  # t * [b, h]
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
        for t in range(T):
            if t == 0:
                rnn_input = esentences
            else:
                rnn_input = rnn_outputs[-1]
            rnn_output, state = cell(rnn_input, state)

            # reuse the scope, very important for custom unfolding
            outer_scope.reuse_variables()

            rnn_outputs.append(rnn_output)

    return tf.stack(rnn_outputs, axis=1), state  # t * [b, h] -> [b, t, h]


def seq2seq_decoder_unfold(self, cell, esentences, initial_state, is_training):
    return _decoder_unfold(cell, esentences, initial_state,
                           self.config.num_steps, is_training)


class BasicSeq2SeqModel(ClippingConstrainLanguageModel):
    ''' Basic Sequence to Sequence (seq2seq) Model
    '''
    build_input = build_static_index_input
    build_embedding = build_embedding_for_index
    rnn_encoder_unfold = lm_dynamic_unfold
    rnn_decoder_unfold = seq2seq_decoder_unfold

    def pool(self, rnn_outputs):
        ''' RNN outputs [b, t, h] -> Sentence embedding [b, h]
        '''
        raise NotImplementedError('BasicSeq2SeqModel.pooler not implemented')

    def build_encoder(self, hinputs, is_training):
        config = self.config
        # encoder cell
        single_cell = self.Cell(config.hidden_size)
        self.equip_delta(single_cell,
                         config.clipping_delta_start,
                         config.clipping_delta_minim,
                         is_training)

        cell, state = prepare_rnn(single_cell,
                                  config.keep_prob,
                                  config.num_layers,
                                  config.batch_size,
                                  is_training)

        # rnn unfolding
        rnn_outputs, state = self.rnn_encoder_unfold(
            cell, hinputs, initial_state=state, is_training=is_training
        )
        # [b, t, h]

        return self.pool(rnn_outputs)  # [b, t, h] -> [b, h]

    def build_decoder(self, esentences, is_training):
        config = self.config
        # decoder cell
        single_cell = self.Cell(config.hidden_size)
        self.equip_delta(single_cell,
                         config.clipping_delta_start,
                         config.clipping_delta_minim,
                         is_training)

        cell, state = prepare_rnn(single_cell,
                                  config.keep_prob,
                                  config.num_layers,
                                  config.batch_size,
                                  is_training)

        # rnn unfolding
        rnn_outputs, state = self.rnn_decoder_unfold(
            cell, esentences, initial_state=state, is_training=is_training
        )
        return rnn_outputs

    def build_flow(self,
                   histories, constraints, targets, weights,
                   is_training):
        config = self.config

        # embedding
        embedding, hinputs = self.build_embedding(histories)

        # apply dropout on input
        if is_training and config.keep_prob < 1:
            hinputs = tf.nn.dropout(hinputs, config.keep_prob)

        esentences = self.build_encoder(hinputs, is_training)
        rnn_outputs = self.build_decoder(esentences, is_training)

        rnn_outputs = tf.reshape(rnn_outputs, [-1, config.hidden_size])
        # [b, t, h] -> [b * t, h]

        # projection
        logits = linear(rnn_outputs,
                        tf.transpose(embedding),
                        config.clipping_gamma)

        return seq2seq_loss(logits, targets, weights)


class LastPoolingSeq2seqModel(BasicSeq2SeqModel):
    ''' Use the last encoder RNN output as the sentence embedding
    '''
    def pool(self, rnn_outputs):
        return rnn_outputs[:, -1, :]


class MaxPoolingSeq2SeqModel(BasicSeq2SeqModel):
    ''' Use max pooling of the encoder RNN outputs as the sentence embedding
    '''
    def pool(self, rnn_outputs):
        return tf.reduce_max(rnn_outputs, 1)
