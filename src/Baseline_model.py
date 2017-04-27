import numpy as np
import tensorflow as tf

from voc import Vocab
from config import WORD_VEC_100

from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl

class Model(object):
    def __init__(self, batch_size=100, vocab_size=5620,
                 word_dim=100, lstm_dim=100, num_classes=4,
                 l2_reg_lambda=0.0,
                 lr=0.001,
                 clip=5,
                 init_embedding=None,
                 bi_gram=False,
                 stack=False,
                 lstm_net=False,
                 bi_direction=False):

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.lstm_dim = lstm_dim
        self.num_classes = num_classes
        self.l2_reg_lambda = l2_reg_lambda
        self.lr = lr
        self.clip = clip
        self.stack = stack
        self.lstm_net = lstm_net
        self.bi_direction = bi_direction

        if init_embedding is None:
            self.init_embedding = np.zeros([vocab_size, word_dim], dtype=np.float32)
        else:
            self.init_embedding = init_embedding

        # placeholders
        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.int32, [None, None])
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.variable_scope("embedding"):
            self.embedding = tf.Variable(
                self.init_embedding,
                name="embedding")

        with tf.variable_scope("softmax"):
            if self.bi_direction:
                self.W = tf.get_variable(
                    shape=[lstm_dim * 2, num_classes],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    name="weights",
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
            else:
                self.W = tf.get_variable(
                    shape=[lstm_dim, num_classes],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    name="weights",
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

            self.b = tf.Variable(
                tf.zeros([num_classes],
                         name="bias"))

        with tf.variable_scope("lstm"):
            if self.lstm_net is False:
                self.fw_cell = core_rnn_cell_impl.GRUCell(self.lstm_dim)
                self.bw_cell = core_rnn_cell_impl.GRUCell(self.lstm_dim)
            else:
                self.fw_cell = core_rnn_cell_impl.BasicLSTMCell(self.lstm_dim)
                self.bw_cell = core_rnn_cell_impl.BasicLSTMCell(self.lstm_dim)

        with tf.variable_scope("forward"):
            seq_len = tf.cast(self.seq_len, tf.int64)
            x = tf.nn.embedding_lookup(self.embedding, self.x)  # batch_size * (sequence*9 or 1) * word_dim
            x = tf.nn.dropout(x, self.dropout_keep_prob)

            size = tf.shape(x)[0]
            if bi_gram is False:
                x = tf.reshape(x, [size, -1, word_dim])  # ba*se*wd
            else:
                x = tf.reshape(x, [size, -1, 9 * word_dim])

            if self.bi_direction:
                (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
                    self.fw_cell,
                    self.bw_cell,
                    x,
                    dtype=tf.float32,
                    sequence_length=seq_len,
                    scope='layer_1'
                )
                output = tf.concat(axis=2, values=[forward_output, backward_output])
                if self.stack:
                    (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
                        self.fw_cell,
                        self.bw_cell,
                        output,
                        dtype=tf.float32,
                        sequence_length=seq_len,
                        scope='layer_2'
                    )
                    output = tf.concat(axis=2, values=[forward_output, backward_output])
            else:
                forward_output,  _ = tf.nn.dynamic_rnn(
                    self.fw_cell,
                    x,
                    dtype=tf.float32,
                    sequence_length=seq_len,
                    scope='layer_1'
                )
                output = forward_output
                if self.stack:
                    forward_output, _ = tf.nn.dynamic_rnn(
                        self.fw_cell,
                        output,
                        dtype=tf.float32,
                        sequence_length=seq_len,
                        scope='layer_2'
                    )
                    output = forward_output

            if self.bi_direction:
                output = tf.reshape(output, [-1, 2 * self.lstm_dim])
            else:
                output = tf.reshape(output, [-1, self.lstm_dim])

            matricized_unary_scores = tf.matmul(output, self.W) + self.b

            self.unary_scores = tf.reshape(
                matricized_unary_scores,
                [size, -1, self.num_classes])

        with tf.variable_scope("loss") as scope:
            # CRF log likelihood
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.unary_scores, self.y, self.seq_len)

            self.loss = tf.reduce_mean(-log_likelihood)

        with tf.variable_scope("train_ops") as scope:
            self.optimizer = tf.train.AdamOptimizer(self.lr)

            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.clip)
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars),
                                                           global_step=self.global_step)

    def train_step(self, sess, x_batch, y_batch, seq_len_batch, dropout_keep_prob):

        feed_dict = {
            self.x: x_batch,
            self.y: y_batch,
            self.seq_len: seq_len_batch,
            self.dropout_keep_prob: dropout_keep_prob
        }
        _, step, loss = sess.run(
            [self.train_op, self.global_step, self.loss],
            feed_dict)

        return step, loss

    def fast_all_predict(self, sess, N, batch_iterator, bigram):
        y_pred, y_true = [], []
        num_batches = int((N - 5) / self.batch_size)

        for i in range(num_batches):

            x_batch, y_batch, seq_len_batch = batch_iterator.next_all_batch(self.batch_size, bigram=bigram)

            # infer predictions
            feed_dict = {
                self.x: x_batch,
                self.y: y_batch,
                self.seq_len: seq_len_batch,
                self.dropout_keep_prob: 1.0
            }

            unary_scores, transition_params = sess.run(
                [self.unary_scores, self.transition_params], feed_dict)

            for unary_scores_, y_, seq_len_ in zip(unary_scores, y_batch, seq_len_batch):
                # remove padding
                unary_scores_ = unary_scores_[:seq_len_]

                # Compute the highest scoring sequence.
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                    unary_scores_, transition_params)

                y_pred += viterbi_sequence
                y_true += y_[:seq_len_].tolist()

        y_true_one, y_pred_one = self.predict(sess, N - self.batch_size * num_batches, batch_iterator)
        y_pred += y_pred_one
        y_true += y_true_one

        return y_true, y_pred

    def predict(self, sess, N, one_iterator):
        y_pred, y_true = [], []
        for i in xrange(N):
            x_one, y_one, len_one = one_iterator.next_pred_one()

            feed_dict = {
                self.x: x_one,
                self.y: y_one,
                self.seq_len: len_one,
                self.dropout_keep_prob: 1.0
            }

            unary_scores, transition_params = sess.run(
                [self.unary_scores, self.transition_params], feed_dict)

            unary_scores_ = unary_scores[0]
            y_one_ = y_one[0]

            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                unary_scores_, transition_params)

            y_pred += viterbi_sequence
            y_true += y_one_[:len_one[0]].tolist()

        return y_true, y_pred

def test():
    init_embedding = Vocab(WORD_VEC_100).word_vectors
    model = Model(2, 5620, 50, 100, 4, init_embedding=init_embedding)
    print model.embedding.get_shape()
    print model.W.get_shape()
    print model.b.get_shape()

    print model.lstm_fw_cell
    print model.lstm_bw_cell

    print model.unary_scores.get_shape()

    print model.loss.get_shape()


if __name__ == "__main__":
    test()