import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, confusion_matrix

from voc import Vocab, OOV, Tag
from config import WORD_VEC_100, DROP_SINGLE, BI_DIRECTION, BI_GRAM, TASK_NAME, STACK_STATUS, LSTM_NET, WORD_SINGLE, TRAIN_PATH

from Baseline_model import Model
import data_helpers
import logging
from prepare_data_index import Data_index

# ==================================================
print 'Generate words and characters need to be trained'
VOCABS = Vocab(WORD_VEC_100, WORD_SINGLE, single_task=True, bi_gram=BI_GRAM, frequency=5)
TAGS = Tag()
init_embedding = VOCABS.word_vectors
da_idx = Data_index(VOCABS, TAGS)
da_idx.process_all_data(BI_GRAM, multitask=False)

tf.flags.DEFINE_integer("vocab_size", init_embedding.shape[0], "vocab_size")

# Data parameters
tf.flags.DEFINE_integer("word_dim", 100, "word_dim")
tf.flags.DEFINE_integer("lstm_dim", 100, "lstm_dim")
tf.flags.DEFINE_integer("num_classes", 4, "num_classes")

# model names
tf.flags.DEFINE_string("model_name", "cws_"+TASK_NAME, "model name")

# Model Hyperparameters[t]
tf.flags.DEFINE_float("lr", 0.01, "learning rate (default: 0.01)")
tf.flags.DEFINE_float("dropout_keep_prob", DROP_SINGLE, "Dropout keep probability (default: 0.8)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.000, "L2 regularizaion lambda (default: 0.5)")
tf.flags.DEFINE_float("clip", 5, "grident clip")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 18000, "Number of training epochs (default: 40)")
tf.flags.DEFINE_integer("evaluate_every", 300, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 300, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("embed_status", True, "embed_status")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log pl:acement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

if FLAGS.embed_status is False:
    init_embedding = None

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items(),reverse=True):
    print("{}={} \n".format(attr.upper(), value))
print("")

logger = logging.getLogger('record_base')
hdlr = logging.FileHandler('Baseline_train.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)


# Load data
print("Loading data...")
if BI_GRAM is False:
    train_file = 'data_' + TASK_NAME + '/train_uni.csv'
    dev_file = 'data_' + TASK_NAME + '/dev_uni.csv'
    test_file = 'data_' + TASK_NAME + '/test_uni.csv'
else:
    train_file = 'data_' + TASK_NAME + '/train.csv'
    dev_file = 'data_' + TASK_NAME + '/dev.csv'
    test_file = 'data_' + TASK_NAME + '/test.csv'

train_df = pd.read_csv(train_file)
train_data_iterator = data_helpers.BucketedDataIterator(train_df)

dev_df = pd.read_csv(dev_file)
dev_data_iterator = data_helpers.BucketedDataIterator(dev_df)

test_df = pd.read_csv(test_file)
test_data_iterator = data_helpers.BucketedDataIterator(test_df)

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)

    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        # build model
        model = Model(batch_size=FLAGS.batch_size,
                      vocab_size=FLAGS.vocab_size,
                      word_dim=FLAGS.word_dim,
                      lstm_dim=FLAGS.lstm_dim,
                      num_classes=FLAGS.num_classes,
                      lr=FLAGS.lr,
                      clip=FLAGS.clip,
                      l2_reg_lambda=FLAGS.l2_reg_lambda,
                      init_embedding=init_embedding,
                      bi_gram=BI_GRAM,
                      stack=STACK_STATUS,
                      lstm_net=LSTM_NET,
                      bi_direction=BI_DIRECTION)

        # Output directory for models
        try:
            shutil.rmtree(os.path.join(os.path.curdir, "models", FLAGS.model_name))
        except:
            pass
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", FLAGS.model_name))
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        filename = 'Base_line_' + str(FLAGS.lr) + '_' + str(FLAGS.dropout_keep_prob) \
        + '_' + str(BI_GRAM) + '_' + str(STACK_STATUS) + '_' + str(LSTM_NET) + '_' + str(BI_DIRECTION)
        checkpoint_prefix = os.path.join(checkpoint_dir, filename)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch, seq_len_batch):
            step, loss = model.train_step(sess,
                x_batch, y_batch, seq_len_batch, FLAGS.dropout_keep_prob)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))

            return step

        def evaluate_word_PRF(y_pred, y, test = False):
            cor_num = 0
            yp_wordnum = y_pred.count(2)+y_pred.count(3)
            yt_wordnum = y.count(2)+y.count(3)
            start = 0
            for i in xrange(len(y)):
                if y[i] == 2 or y[i] == 3:
                    flag = True
                    for j in xrange(start, i+1):
                        if y[j] != y_pred[j]:
                            flag = False
                    if flag == True:
                        cor_num += 1
                    start = i+1

            P = cor_num / float(yp_wordnum)
            R = cor_num / float(yt_wordnum)
            F = 2 * P * R / (P + R)
            print 'P: ', P
            print 'R: ', R
            print 'F: ', F
            if test:
                return P,R,F
            else:
                return F

        def final_test_step(df, iterator, test=False, bigram=False):
            N = df.shape[0]
            y_true, y_pred = model.fast_all_predict(sess, N, iterator, bigram=bigram)
            if test:
                print 'Test:'
            else:
                print 'Dev'
            return y_pred, y_true

        # train loop
        logger.info('Task_{} Training starts'.format(TASK_NAME))
        best_accuary = 0.0
        best_step = 0
        p, r, f = 0.0, 0.0, 0.0
        for i in range(FLAGS.num_epochs):
            x_batch, y_batch, seq_len_batch = train_data_iterator.next_batch(FLAGS.batch_size, bigram=BI_GRAM)

            current_step = train_step(x_batch, y_batch, seq_len_batch)

            if current_step % FLAGS.evaluate_every == 0:
                yp,yt = final_test_step(dev_df, dev_data_iterator, bigram=BI_GRAM)
                tmpacc = evaluate_word_PRF(yp,yt)
                if best_accuary < tmpacc:
                    best_accuary = tmpacc
                    best_step = current_step
                    yp_test, yt_test = final_test_step(test_df, test_data_iterator, test=True, bigram=BI_GRAM)
                    p, r, f = evaluate_word_PRF(yp_test, yt_test, test=True)
                    path = saver.save(sess, checkpoint_prefix)
                    print("Saved model checkpoint to {}\n".format(path))

                if current_step - best_step > 2000:
                    print FLAGS.model_name, 'dropout:',FLAGS.dropout_keep_prob
                    print("Dev acc is not getting better in 2000 steps, triggers normal early stop")
                    break

        logger.info('-------------Show the results:{}--------------'.format(filename))
        logger.info('P:{:.2f},R:{:.2f},F:{:.2f},step:{},'.format(100 * p, 100 * r, 100 * f, best_step))
        saver.restore(sess, path)
        yp, yt = final_test_step(test_df, test_data_iterator, test=True, bigram=BI_GRAM)
        evaluate_word_PRF(yp, yt)
        gold_path = os.path.join(os.path.curdir, 'data_'+TASK_NAME, 'test_gold')
        dict_path = os.path.join(os.path.curdir, 'data_'+TASK_NAME, 'words')
        test_path = os.path.join(os.path.curdir, 'data_'+TASK_NAME, 'test')
        dest_path = os.path.join(os.path.curdir, 'data_'+TASK_NAME, filename)

        tmpOOV = OOV(goldpath=gold_path, dictpath=dict_path, testpath=test_path, destpath=dest_path,
                     yp=yp)
        oovrate = tmpOOV.eval_oov_rate()
        logger.info('OOV:{:.2f}\n'.format(100 * oovrate))
        logger.info('--------------Train ends-------------')