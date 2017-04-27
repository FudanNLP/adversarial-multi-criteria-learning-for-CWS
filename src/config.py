# -*- coding: utf-8 -*-

import os

#Multi_Model
MODEL_TYPE = 'Model1'
ADV_STATUS = False
DROP_OUT = [0.78, 0.83, 0.65, 0.6, 0.7, 0.5, 0.65, 0.5, 0.5]
CORPUS = 9

#Baseline
DROP_SINGLE = 0.5
LSTM_NET = True
STACK_STATUS = False
BI_DIRECTION = True
BI_GRAM = True
TASK_NAME = 'sxu'

DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, 'data_'+TASK_NAME)
MODEL_DIR = os.path.join(DIR, 'models')

WORD_VEC_100 = os.path.join(MODEL_DIR, 'vec100.txt')

TRAIN_PATH = os.path.join(DATADIR, 'train')
DEV_PATH = os.path.join(DATADIR, 'dev')
TEST_PATH = os.path.join(DATADIR, 'test')
WORD_SINGLE = os.path.join(DATADIR, 'words_for_training')

WORD_DICT = os.path.join(MODEL_DIR, 'train_words')
#This is used for generating multi_task train, dev, test input
TRAIN_DATA_MT = os.path.join(DATADIR, 'train_mt.csv')
DEV_DATA_MT = os.path.join(DATADIR, 'dev_mt.csv')
TEST_DATA_MT = os.path.join(DATADIR, 'test_mt.csv')

TRAIN_DATA_UNI = os.path.join(DATADIR, 'train_uni.csv')
DEV_DATA_UNI = os.path.join(DATADIR, 'dev_uni.csv')
TEST_DATA_UNI = os.path.join(DATADIR, 'test_uni.csv')

TRAIN_DATA_BI = os.path.join(DATADIR, 'train.csv')
DEV_DATA_BI = os.path.join(DATADIR, 'dev.csv')
TEST_DATA_BI = os.path.join(DATADIR, 'test.csv')

TRAIN_FILE = ['data_msr/train_mt.csv','data_as/train_mt.csv','data_pku/train_mt.csv','data_ctb/train_mt.csv',
              'data_ckip/train_mt.csv','data_cityu/train_mt.csv','data_ncc/train_mt.csv','data_sxu/train_mt.csv','data_weibo/train_mt.csv']
DEV_FILE = ['data_msr/dev_mt.csv','data_as/dev_mt.csv','data_pku/dev_mt.csv','data_ctb/dev_mt.csv',
            'data_ckip/dev_mt.csv','data_cityu/dev_mt.csv','data_ncc/dev_mt.csv','data_sxu/dev_mt.csv','data_weibo/dev_mt.csv']
TEST_FILE = ['data_msr/test_mt.csv','data_as/test_mt.csv','data_pku/test_mt.csv','data_ctb/test_mt.csv',
             'data_ckip/test_mt.csv','data_cityu/test_mt.csv','data_ncc/test_mt.csv','data_sxu/test_mt.csv','data_weibo/test_mt.csv']

DATA_FILE = ['data_msr','data_as','data_pku','data_ctb','data_ckip','data_cityu','data_ncc','data_sxu', 'data_weibo']
MAX_LEN = 80

