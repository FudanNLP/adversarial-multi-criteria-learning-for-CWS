import numpy as np
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
import tensorflow as tf
import math

from voc import Vocab
from config import WORD_VEC_100
"""
    Build the multi_task_cws model.
    Args:
      num_corpus: int, The number of corpus used in multi_task.
      adv: boolean, If True, adversarial is added in the model.
      gates: If True, gate is added between shared layer and private layer.
             we didn't add gate in this paper
      reuseshare: If True, the output of shared layer will be used as part of the final output
      sep: If True, the output of the shared layer will not be used as part of the input of private layer
      adv_weight: float, the weight of adversarial loss in the combined loss(cws_loss + hess_loss * weight)
"""
class MultiModel(object):
    def __init__(self, batch_size=100, vocab_size=5620,
                 word_dim=100, lstm_dim=100, num_classes=4,
                 num_corpus = 8,
                 l2_reg_lambda=0.0,
                 adv_weight = 0.05,
                 lr=0.001,
                 clip=5,
                 init_embedding=None,
                 gates=False,
                 adv=False,
                 reuseshare=False,
                 sep=False):

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.lstm_dim = lstm_dim
        self.num_classes = num_classes
        self.num_corpus = num_corpus
        self.l2_reg_lambda = l2_reg_lambda
        self.lr = lr
        self.clip = clip
        self.gateus = gates
        self.adv = adv
        self.reuse = reuseshare


        # placeholders
        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.int32, [None, None])
        self.y_class = tf.placeholder(tf.int32, [None])
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        if init_embedding is None:
            self.init_embedding = np.zeros([vocab_size, word_dim], dtype=np.float32)
        else:
            self.init_embedding = init_embedding

        with tf.variable_scope("embedding") as scope:
            self.embedding = tf.Variable(
                self.init_embedding,
                name="embedding")

        lstm_fw_cell = core_rnn_cell_impl.BasicLSTMCell(self.lstm_dim)
        lstm_bw_cell = core_rnn_cell_impl.BasicLSTMCell(self.lstm_dim)

        def _shared_layer(input_data, seq_len):

            (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell,
                lstm_bw_cell,
                input_data,
                dtype=tf.float32,
                sequence_length=seq_len,
            )
            output = tf.concat(axis=2, values=[forward_output, backward_output])

            return output

        def _private_layer(output_pub, input_data, seq_len, y):
            size = tf.shape(input_data)[0]
            if sep is False:
                if self.gateus:
                    target_dim = tf.shape(output_pub)[2]
                    factor = tf.concat(axis=2, values=[input_data, output_pub])
                    dim = tf.shape(factor)[2]
                    W_g = tf.get_variable(shape=[self.lstm_dim * 2 + self.word_dim * 9, self.lstm_dim * 2],
                        initializer=tf.truncated_normal_initializer(stddev=0.01),
                        name="w_gates")
                    factor = tf.reshape(factor, [-1, dim])
                    gate = tf.matmul(factor, W_g)

                    output_prep = tf.nn.sigmoid(tf.reshape(output_pub,[-1, target_dim]))
                    output_pub = tf.multiply(output_prep, gate)
                    output_pub = tf.reshape(output_pub, [size, -1, target_dim])

                combined_input_data = tf.concat(axis=2, values=[input_data, output_pub])
                combined_input_data = tf.reshape(combined_input_data, [size, -1, self.lstm_dim * 2 + self.word_dim * 9])
            else:
                combined_input_data = input_data

            (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell,
                lstm_bw_cell,
                combined_input_data,
                dtype=tf.float32,
                sequence_length=seq_len,
            )
            output = tf.concat(axis=2, values=[forward_output, backward_output])

            if self.reuse is False:
                output = tf.reshape(output, [-1, self.lstm_dim * 2])
                W = tf.get_variable(
                    shape=[lstm_dim * 2, num_classes],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    name="weights",
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
            else:
                output = tf.concat(axis=2, values=[output, output_pub])
                output = tf.reshape(output, [-1, self.lstm_dim * 4])
                W = tf.get_variable(
                    shape=[lstm_dim * 4, num_classes],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    name="weights",
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

            b = tf.Variable(tf.zeros([num_classes], name="bias"))

            matricized_unary_scores = tf.matmul(output, W) + b
            unary_scores = tf.reshape(
                matricized_unary_scores,
                [size, -1, self.num_classes])

            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                unary_scores, y, self.seq_len)

            if self.gateus:
                return unary_scores, log_likelihood, transition_params, gate
            else:
                return unary_scores, log_likelihood, transition_params

        #domain layer
        def _domain_layer(output_pub, seq_len):  #output_pub batch_size * seq_len * (2 * lstm_dim)
            W_classifier = tf.get_variable(shape=[2 * lstm_dim, num_corpus],
                                           initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(num_corpus))),
                                           name = 'W_classifier')
            bias = tf.Variable(
                tf.zeros([num_corpus],
                         name="class_bias"))
            output_avg = reduce_avg(output_pub, seq_len, 1)  #output_avg batch_size * (2 * lstm_dim)
            logits = tf.matmul(output_avg, W_classifier) + bias   #logits batch_size * num_corpus
            return logits

        def _Hloss(logits):
            log_soft = tf.nn.log_softmax(logits)  # batch_size * num_corpus
            soft = tf.nn.softmax(logits)
            H_mid = tf.reduce_mean(tf.multiply(soft, log_soft), axis=0)  # [num_corpus]
            H_loss = tf.reduce_sum(H_mid)
            return H_loss


        def _Dloss(logits, y_class):
            labels = tf.to_int64(y_class)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='xentropy')
            D_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            return D_loss

        def _loss(log_likelihood):

            loss = tf.reduce_mean(-log_likelihood)

            return loss

        def _training(loss):

            optimizer = tf.train.AdamOptimizer(self.lr)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.clip)
            train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                      global_step = global_step)

            return train_op, global_step

        def _trainingPrivate(loss, taskid):
            optimizer = tf.train.AdamOptimizer(self.lr)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=taskid)
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.clip)
            train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                 global_step=global_step)

            return train_op, global_step

        def _trainingDomain(loss):
            optimizer = tf.train.AdamOptimizer(self.lr)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='domain')
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.clip)
            train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                 global_step=global_step)

            return train_op, global_step

        def _trainingShared(loss, taskid):
            optimizer = tf.train.AdamOptimizer(self.lr)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=taskid) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='embedding')
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.clip)
            train_op = optimizer.apply_gradients(zip(grads, tvars),
                                      global_step=global_step)

            return train_op, global_step

        seq_len = tf.cast(self.seq_len, tf.int64)
        x = tf.nn.embedding_lookup(self.embedding, self.x)  # batch_size * (sequence*9) * word_dim
        size = tf.shape(x)[0]
        # we use window_size 5 and bi_gram, which means for each position,
        # there will be 5+4=9 (character or word) features
        x = tf.reshape(x, [size, -1, 9 * word_dim])  # ba*se*(9*wd)
        x = tf.nn.dropout(x, self.dropout_keep_prob)
        #task1:msr 2:as 3 pku 4 ctb 5 ckip 6 cityu 7 ncc 8 sxu 9 weibo
        with tf.variable_scope("shared"):
            output_pub = _shared_layer(x, seq_len)

        #add adverisal op
        if self.adv:
            with tf.variable_scope("domain"):
                logits = _domain_layer(output_pub, seq_len)
            self.H_loss = _Hloss(logits)
            self.D_loss = _Dloss(logits, self.y_class)

        self.scores = []
        self.transition = []
        self.gate = []
        loglike = []
        #add task op
        for i in range(1, self.num_corpus+1):
            Taskid = 'task' + str(i)
            with tf.variable_scope(Taskid):
                condition = _private_layer(output_pub, x, seq_len, self.y)
                self.scores.append(condition[0])
                loglike.append(condition[1])
                self.transition.append(condition[2])
                if self.gateus:
                    self.gate.append(condition[3])

        #loss_com is combination loss(cws + hess), losses is basic loss(cws)
        self.losses = [_loss(o) for o in loglike]
        if self.adv:
            self.loss_com = [adv_weight * self.H_loss + o for o in self.losses]
            self.domain_op, self.global_step_domain = _trainingDomain(self.D_loss)
        #task_basic_op is for basic train
        self.task_basic_op = []
        self.global_basic_step = []
        for i in range(1, self.num_corpus+1):
            res = _training(self.losses[i-1])
            self.task_basic_op.append(res[0])
            self.global_basic_step.append(res[1])

        #task_op is for combination train(cws_loss + hess_loss * adv_weight)
        if self.adv:
            self.task_op = []
            self.global_step = []
            for i in range(1, self.num_corpus+1):
                Taskid = 'task' + str(i)
                res = _trainingShared(self.loss_com[i-1], taskid=Taskid)
                self.task_op.append(res[0])
                self.global_step.append(res[1])

        #task_op_ss is for private train
        self.task_op_ss = []
        self.global_pristep = []
        for i in range(1, self.num_corpus+1):
            Taskid = 'task' + str(i)
            res = _trainingPrivate(self.losses[i-1], Taskid)
            self.task_op_ss.append(res[0])
            self.global_pristep.append(res[1])

    #train all the basic model cwsloss, all parameters
    def train_step_basic(self, sess, x_batch, y_batch, seq_len_batch, dropout_keep_prob, task_op, global_step, loss):

        feed_dict = {
            self.x: x_batch,
            self.y: y_batch,
            self.seq_len: seq_len_batch,
            self.dropout_keep_prob: dropout_keep_prob
        }
        _, step, loss = sess.run(
            [task_op, global_step, loss],
            feed_dict)

        return step, loss

    # train all the cwsloss + hesloss VS advloss, main_line parameters Or cwsloss VS advloss(depends on taskop_type)
    def train_step_task(self, sess, x_batch, y_batch, seq_len_batch, y_class_batch, dropout_keep_prob, task_op, global_step, loss,domain_op, global_step_domain, Dloss, Hloss):

        feed_dict = {
            self.x: x_batch,
            self.y: y_batch,
            self.y_class: y_class_batch,
            self.seq_len: seq_len_batch,
            self.dropout_keep_prob: dropout_keep_prob
        }
        _, step_norm, loss_norm, _v, step_adv, loss_adv, loss_hess = sess.run(
            [task_op, global_step, loss, domain_op, global_step_domain, Dloss, Hloss],
            feed_dict)
        return step_norm, loss_norm, loss_adv, loss_hess


    # train only the private params, cwsloss
    def train_step_pritask(self, sess, x_batch, y_batch, seq_len_batch, dropout_keep_prob, task_op, global_step, loss):
        feed_dict = {
            self.x: x_batch,
            self.y: y_batch,
            self.seq_len: seq_len_batch,
            self.dropout_keep_prob: dropout_keep_prob
        }
        _, step, loss = sess.run(
            [task_op, global_step, loss],
            feed_dict)

        return step, loss


    #predict all for tasks
    def fast_all_predict(self, sess, N, batch_iterator, scores, transition_param):
        y_pred, y_true = [], []
        num_batches = int((N - 1) / self.batch_size)
        for i in range(num_batches):

            x_batch, y_batch, seq_len_batch = batch_iterator.next_all_batch(self.batch_size)

            # infer predictions
            feed_dict = {
                self.x: x_batch,
                self.y: y_batch,
                self.seq_len: seq_len_batch,
                self.dropout_keep_prob: 1.0
            }

            unary_scores, transition_params = sess.run(
                    [scores, transition_param], feed_dict)

            for unary_scores_, y_, seq_len_ in zip(unary_scores, y_batch, seq_len_batch):
                # remove padding
                unary_scores_ = unary_scores_[:seq_len_]

                # Compute the highest scoring sequence.
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                    unary_scores_, transition_params)

                y_pred += viterbi_sequence
                y_true += y_[:seq_len_].tolist()

        y_true_one, y_pred_one = self.predict(sess, N - self.batch_size * num_batches, batch_iterator, scores, transition_param)
        y_pred += y_pred_one
        y_true += y_true_one

        return y_true, y_pred

    # predict one by one for tasks
    def predict(self, sess, N, one_iterator, scores, transition_param):
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
                [scores, transition_param], feed_dict)

            unary_scores_ = unary_scores[0]
            y_one_ = y_one[0]

            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                unary_scores_, transition_params)

            y_pred += viterbi_sequence
            y_true += y_one_[:len_one[0]].tolist()

        return y_true, y_pred


def mkMask(input_tensor, maxLen):
    shape_of_input = tf.shape(input_tensor)
    shape_of_output = tf.concat(axis=0, values=[shape_of_input, [maxLen]])

    oneDtensor = tf.reshape(input_tensor, shape=(-1,))
    flat_mask = tf.sequence_mask(oneDtensor, maxlen=maxLen)
    return tf.reshape(flat_mask, shape_of_output)


def reduce_avg(reduce_target, lengths, dim):
    """
    Args:
        reduce_target : shape(d_0, d_1,..,d_dim, .., d_k)
        lengths : shape(d0, .., d_(dim-1))
        dim : which dimension to average, should be a python number
    """
    shape_of_lengths = lengths.get_shape()
    shape_of_target = reduce_target.get_shape()
    if len(shape_of_lengths) != dim:
        raise ValueError(('Second input tensor should be rank %d, ' +
                         'while it got rank %d') % (dim, len(shape_of_lengths)))
    if len(shape_of_target) < dim+1 :
        raise ValueError(('First input tensor should be at least rank %d, ' +
                         'while it got rank %d') % (dim+1, len(shape_of_target)))

    rank_diff = len(shape_of_target) - len(shape_of_lengths) - 1
    mxlen = tf.shape(reduce_target)[dim]
    mask = mkMask(lengths, mxlen)
    if rank_diff!=0:
        len_shape = tf.concat(axis=0, values=[tf.shape(lengths), [1]*rank_diff])
        mask_shape = tf.concat(axis=0, values=[tf.shape(mask), [1]*rank_diff])
    else:
        len_shape = tf.shape(lengths)
        mask_shape = tf.shape(mask)
    lengths_reshape = tf.reshape(lengths, shape=len_shape)
    mask = tf.reshape(mask, shape=mask_shape)

    mask_target = reduce_target * tf.cast(mask, dtype=reduce_target.dtype)

    red_sum = tf.reduce_sum(mask_target, axis=[dim], keep_dims=False)
    red_avg = red_sum / (tf.to_float(lengths_reshape) + 1e-30)
    # red_avg = red_sum / lengths_reshape
    return red_avg