import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import sys

from sklearn.metrics import accuracy_score

from voc import Vocab, OOV
from config import WORD_VEC_100, TRAIN_FILE, TEST_FILE, DEV_FILE, DATA_FILE, DROP_OUT, WORD_DICT, MODEL_TYPE, ADV_STATUS


from AdvMulti_model import MultiModel
import data_helpers

# ==================================================

init_embedding = Vocab(WORD_VEC_100, WORD_DICT, single_task=False, bi_gram=True).word_vectors
tf.flags.DEFINE_integer("vocab_size", init_embedding.shape[0], "vocab_size")

# Data parameters
tf.flags.DEFINE_integer("word_dim", 100, "word_dim")
tf.flags.DEFINE_integer("lstm_dim", 100, "lstm_dim")
tf.flags.DEFINE_integer("num_classes", 4, "num_classes")
tf.flags.DEFINE_integer("num_corpus", 9, "num_corpus")
tf.flags.DEFINE_boolean("embed_status", True, "gate_status")
tf.flags.DEFINE_boolean("gate_status", False, "gate_status")
tf.flags.DEFINE_boolean("real_status", True, "real_status")
tf.flags.DEFINE_boolean("train", True, "train_status")

# Model Hyperparameters[t]
tf.flags.DEFINE_float("lr", 0.01, "learning rate (default: 0.01)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.000, "L2 regularizaion lambda (default: 0.5)")
tf.flags.DEFINE_float("adv_weight", 0.06, "L2 regularizaion lambda (default: 0.5)")
tf.flags.DEFINE_float("clip", 5, "gradient clip")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("batch_size_big", 256, "Batch Size Big(default: 64)")
tf.flags.DEFINE_integer("batch_size_huge", 512, "Batch Size Huge(default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2400, "Number of training epochs (default: 40)")
tf.flags.DEFINE_integer("num_epochs_private", 24000, "Number of training epochs (default: 40)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

if FLAGS.embed_status is False:
    init_embedding = None

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items(),reverse=True):
    print("{}={} \n".format(attr.upper(), value))
print("")
#define log file
logger = logging.getLogger('record')
hdlr = logging.FileHandler('AdvMulti_train.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

if MODEL_TYPE == 'Model1':
    reuse_status = True
    sep_status = True
elif MODEL_TYPE == 'Model2':
    reuse_status = False
    sep_status = False
elif MODEL_TYPE == 'Model3':
    reuse_status = True
    sep_status = False
else:
    print 'choose the correct multi_model, the listed choices are Model1, Model2, Model3'
    logger.warn('Wrong Model Choosen {}'.format(MODEL_TYPE))
    sys.exit()

stats = [FLAGS.embed_status, FLAGS.gate_status, ADV_STATUS]
posfix = map(lambda x: 'Y' if x else 'N', stats)
posfix.append(MODEL_TYPE)
if ADV_STATUS:
    posfix.append(str(FLAGS.adv_weight))

#Load data
train_data_iterator = []
dev_data_iterator = []
test_data_iterator = []
dev_df = []
test_df = []
print("Loading data...")
for i in xrange(FLAGS.num_corpus):
    train_data_iterator.append(data_helpers.BucketedDataIterator(pd.read_csv(TRAIN_FILE[i])))
    dev_df.append(pd.read_csv(DEV_FILE[i]))
    dev_data_iterator.append(data_helpers.BucketedDataIterator(dev_df[i]))
    test_df.append(pd.read_csv(TEST_FILE[i]))
    test_data_iterator.append(data_helpers.BucketedDataIterator(test_df[i]))

logger.info('-'*50)

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
        model = MultiModel(batch_size=FLAGS.batch_size,
                      vocab_size=FLAGS.vocab_size,
                      word_dim=FLAGS.word_dim,
                      lstm_dim=FLAGS.lstm_dim,
                      num_classes=FLAGS.num_classes,
                      num_corpus=FLAGS.num_corpus,
                      lr=FLAGS.lr,
                      clip=FLAGS.clip,
                      l2_reg_lambda=FLAGS.l2_reg_lambda,
                      adv_weight = FLAGS.adv_weight,
                      init_embedding=init_embedding,
                      gates=FLAGS.gate_status,
                      adv=ADV_STATUS,
                      reuseshare=reuse_status,
                      sep=sep_status)

        # Output directory for models
        model_name = 'multi_model'+ str(FLAGS.num_corpus)
        try:
            shutil.rmtree(os.path.join(os.path.curdir, "models", model_name))
        except:
            pass
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", model_name))
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        # modeli_embed_adv_gate_diff_dropout
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_all = []
        for i in xrange(1, FLAGS.num_corpus+1):
            filename = 'task' + str(i) + '_' + '_'.join(posfix)
            checkpoint_all.append(os.path.join(checkpoint_dir, filename))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=20)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Initialize all the op
        # basictask is for basic cws loss, task is for combination loss, privatetask is for cws loss only on solo params
        # testTask is for prediction, taskID is for storage of loading from .csv
        basictask = []
        task = []
        privateTask = []
        testTask = []
        taskID = []
        for i in xrange(FLAGS.num_corpus):
            basictask.append([model.task_basic_op[i],model.global_basic_step[i],model.losses[i]])
            if model.adv:
                task.append([model.task_op[i], model.global_step[i], model.loss_com[i]])
            privateTask.append([model.task_op_ss[i], model.global_pristep[i], model.losses[i]])
            testTask.append([model.scores[i], model.transition[i]])
            taskID.append([train_data_iterator[i],dev_df[i],dev_data_iterator[i],test_df[i],test_data_iterator[i]])

        def train_step_basic(x_batch, y_batch, seq_len_batch, id):
            step, loss = model.train_step_basic(sess,
                x_batch, y_batch, seq_len_batch, DROP_OUT[id-1], basictask[id-1][0], basictask[id-1][1], basictask[id-1][2])

            time_str = datetime.datetime.now().isoformat()
            print("Task_{}: {}: step {}, loss {:g}".format(id, time_str, step, loss))

            return step


        def train_step_all(x_batch, y_batch, y_class_batch, seq_len_batch, id):
            step, loss_cws, loss_adv, loss_hess = model.train_step_task(sess,
                   x_batch, y_batch, seq_len_batch, y_class_batch, DROP_OUT[id-1], task[id-1][0], task[id-1][1], task[id-1][2], model.domain_op, model.global_step_domain, model.D_loss, model.H_loss)

            time_str = datetime.datetime.now().isoformat()
            print("Task_{}: {}: step {}, loss_cws {:g}, loss_adv {:g}, loss_hess {:g}".format(id, time_str, step, loss_cws, loss_adv, loss_hess))

            return step

        def train_step_private(x_batch, y_batch, seq_len_batch, id):
            step, loss = model.train_step_pritask(sess,
                x_batch, y_batch, seq_len_batch, DROP_OUT[id-1], privateTask[id-1][0], privateTask[id-1][1], privateTask[id-1][2])

            time_str = datetime.datetime.now().isoformat()
            print("Task_{}: {}: step {}, loss {:g}".format(id, time_str, step, loss))

            return step

        def evaluate_word_PRF(y_pred, y, test=False):
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
                    if flag:
                        cor_num += 1
                    start = i+1

            P = cor_num / float(yp_wordnum)
            R = cor_num / float(yt_wordnum)
            F = 2 * P * R / (P + R)
            print 'P: ', P
            print 'R: ', R
            print 'F: ', F
            if test:
                return P, R, F
            else:
                return F

        def final_test_step(df, iterator, idx, test=False):
            N = df.shape[0]
            y_true, y_pred = model.fast_all_predict(sess, N, iterator, testTask[idx-1][0],testTask[idx-1][1])
            if test:
                print "test:"
            else:
                print "dev:"
            return y_pred, y_true

        # train loop
        if FLAGS.train:
            best_accuary = [0.0] * FLAGS.num_corpus
            best_step_all = [0] * FLAGS.num_corpus
            best_pval = [0.0] * FLAGS.num_corpus
            best_rval = [0.0] * FLAGS.num_corpus
            best_fval = [0.0] * FLAGS.num_corpus
            best_step_private = [0] * FLAGS.num_corpus
            flag = [False] * FLAGS.num_corpus
            logger.info('-------------Public train starts--------------')
            for i in range(FLAGS.num_epochs):
                for j in range(1, FLAGS.num_corpus + 1):
                    if model.adv:
                        if j == 1:
                            x_batch, y_batch, y_class, seq_len_batch = taskID[j - 1][0].next_batch(FLAGS.batch_size_big, round=j-1, classifier=True)
                        elif j == 2:
                            x_batch, y_batch, y_class, seq_len_batch = taskID[j - 1][0].next_batch(FLAGS.batch_size_huge, round=j-1, classifier=True)
                        else:
                            x_batch, y_batch, y_class, seq_len_batch = taskID[j - 1][0].next_batch(FLAGS.batch_size, round=j-1, classifier=True)
                        current_step = train_step_all(x_batch, y_batch, y_class, seq_len_batch, j)
                    else:
                        if j == 1:
                            x_batch, y_batch, seq_len_batch = taskID[j - 1][0].next_batch(FLAGS.batch_size_big)
                        elif j == 2:
                            x_batch, y_batch, seq_len_batch = taskID[j - 1][0].next_batch(FLAGS.batch_size_huge)
                        else:
                            x_batch, y_batch, seq_len_batch = taskID[j - 1][0].next_batch(FLAGS.batch_size)
                        current_step = train_step_basic(x_batch, y_batch, seq_len_batch, j)

                    if current_step % FLAGS.evaluate_every == 0:
                        yp,yt = final_test_step(taskID[j-1][1], taskID[j-1][2], j)
                        tmp_f = evaluate_word_PRF(yp,yt)
                        if best_accuary[j - 1] < tmp_f:
                            best_accuary[j - 1] = tmp_f
                            best_step_all[j - 1] = current_step
                            if FLAGS.real_status:
                                path = saver.save(sess, checkpoint_all[j-1])
                                print("Saved model checkpoint to {}\n".format(path))
                            else:
                                print("This is only for trial and error\n")
                            yp_test, yt_test = final_test_step(taskID[j-1][3], taskID[j-1][4], j, test=True)
                            pval, rval, fval = evaluate_word_PRF(yp_test, yt_test, test=True)
                            best_pval[j - 1] = pval
                            best_rval[j - 1] = rval
                            best_fval[j - 1] = fval
            logger.info('-----------Public train ends-------------')
            for i in xrange(FLAGS.num_corpus):
                logger.info('Task{} best step is {} and p:{:.2f} r:{:.2f} f:{:.2f}'.format(i + 1, best_step_all[i],best_pval[i]*100, best_rval[i]*100,best_fval[i]*100))

            for i in range(FLAGS.num_epochs_private):
                stop = True
                for j in range(FLAGS.num_corpus):
                    if flag[j] is False:
                        stop = False
                if stop is False:
                    for j in range(1, FLAGS.num_corpus + 1):
                        if flag[j - 1]:
                            continue
                        else:
                            x_batch, y_batch, seq_len_batch = taskID[j - 1][0].next_batch(FLAGS.batch_size)
                            current_step = train_step_private(x_batch, y_batch, seq_len_batch, j)
                            if current_step % FLAGS.evaluate_every == 0:
                                yp, yt = final_test_step(taskID[j - 1][1], taskID[j - 1][2], j)
                                tmp_f = evaluate_word_PRF(yp, yt)
                                if best_accuary[j - 1] < tmp_f:
                                    best_accuary[j - 1] = tmp_f
                                    best_step_private[j - 1] = current_step
                                    if FLAGS.real_status:
                                        path = saver.save(sess, checkpoint_all[j - 1])
                                        print("Saved model checkpoint to {}\n".format(path))
                                    else:
                                        print("This is only for trial and error\n")
                                    yp_test, yt_test = final_test_step(taskID[j-1][3], taskID[j-1][4], j, test=True)
                                    best_pval[j - 1], best_rval[j-1], best_fval[j-1] = evaluate_word_PRF(yp_test, yt_test, test=True)
                                elif current_step - best_step_private[j - 1] > 2000:
                                    print("Task_{} didn't get better results in more than 2000 steps".format(j))
                                    flag[j - 1] = True
                else:
                    print 'Early stop triggered, all the tasks have been finished. Dropout:', DROP_OUT
                    break


        if FLAGS.real_status:
            logger.info('-------------Show the results------------')
            for i in xrange(FLAGS.num_corpus):
                filename = 'Model' + str(i+1) + '_' + '_'.join(posfix)
                saver.restore(sess, checkpoint_all[i])
                print 'Task:{}\n'.format(i+1)
                logger.info('Task:{}, filename:{}'.format(i+1, filename))
                yp, yt = final_test_step(taskID[i][3], taskID[i][4], i+1, test=True)
                evaluate_word_PRF(yp, yt)
                gold_path = os.path.join(os.path.curdir, DATA_FILE[i], 'test_gold')
                dict_path = os.path.join(os.path.curdir, DATA_FILE[i], 'words')
                test_path = os.path.join(os.path.curdir, DATA_FILE[i], 'test')
                dest_path = os.path.join(os.path.curdir, DATA_FILE[i], 'dest_gold'+filename)

                tmpOOV = OOV(goldpath=gold_path, dictpath=dict_path, testpath=test_path, destpath=dest_path, yp=yp)
                oovrate = tmpOOV.eval_oov_rate()
                logger.info('Task{} best step is {} and p:{:.2f} r:{:.2f} f:{:.2f} oov:{:.2f}'.format(i+1, best_step_private[i], best_pval[i]*100, best_rval[i]*100, best_fval[i]*100, oovrate*100))


