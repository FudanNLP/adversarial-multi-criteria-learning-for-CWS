# -*- coding: utf-8 -*-
import csv

from config import MAX_LEN, WORD_VEC_100
from config import TRAIN_DATA_MT, TRAIN_PATH, DEV_DATA_MT, DEV_PATH, TEST_DATA_MT, TEST_PATH, WORD_DICT
from config import TRAIN_DATA_BI, DEV_DATA_BI, TEST_DATA_BI, TRAIN_DATA_UNI, DEV_DATA_UNI, TEST_DATA_UNI, WORD_SINGLE
from voc import Vocab, Tag
import argparse


class Data_index(object):
    def __init__(self, Vocabs, tags):
        self.VOCABS = Vocabs
        self.TAGS = tags

    def to_index_bi(self, words, tags):
        word_idx = []
        words.append('<EOS>')
        words.append('<EOS>')
        words.insert(0, '<BOS>')
        words.insert(0, '<BOS>')
        for i in xrange(2, len(words)-2):
            for j in xrange(-2,3):
                if words[i+j] in self.VOCABS.word2idx:
                    word_idx.append(self.VOCABS.word2idx[words[i+j]])
                else:
                    word_idx.append(self.VOCABS.word2idx['<OOV>'])
            for j in xrange(-2,2):
                if words[i+j]+words[i+j+1] in self.VOCABS.word2idx:
                    word_idx.append(self.VOCABS.word2idx[words[i+j]+words[i+j+1]])
                else:
                    word_idx.append(self.VOCABS.word2idx['<OOV>'])

        tag_idx = [self.TAGS.tag2idx[tag] for tag in tags]

        return ','.join(map(str, word_idx)), ','.join(map(str, tag_idx))

    def to_index(self, words, tags):
        word_idx = []
        for word in words:
            if word in self.VOCABS.word2idx:
                word_idx.append(self.VOCABS.word2idx[word])
            else:
                word_idx.append(self.VOCABS.word2idx['<OOV>'])

        tag_idx = [self.TAGS.tag2idx[tag] for tag in tags]

        return ','.join(map(str, word_idx)), ','.join(map(str, tag_idx))

    def process_file(self, path, output, bigram=False):
        src_data, data, label = self.process_data(path)
        for words,tags in zip(data,label):
            length = len(words)
            ratio = (length-1)/MAX_LEN
            for i in xrange(0, ratio+1):
                tmpwords = words[MAX_LEN*i:MAX_LEN*(i+1)]
                tmptags = tags[MAX_LEN*i:MAX_LEN*(i+1)]
                if bigram:
                    word_idx, tag_idx = self.to_index_bi(tmpwords, tmptags)
                    length = len(tmpwords) - 4
                else:
                    word_idx, tag_idx = self.to_index(tmpwords, tmptags)
                    length = len(tmpwords)
                output.writerow([word_idx, tag_idx, length])

    def process_all_data(self, bigram=False, multitask=False):
        if bigram is False:
            f_train = open(TRAIN_DATA_UNI, 'w')
            f_dev = open(DEV_DATA_UNI, 'w')
            f_test = open(TEST_DATA_UNI, 'w')
        elif multitask:
            f_train = open(TRAIN_DATA_MT, 'w')
            f_dev = open(DEV_DATA_MT, 'w')
            f_test = open(TEST_DATA_MT, 'w')
        else:
            f_train = open(TRAIN_DATA_BI, 'w')
            f_dev = open(DEV_DATA_BI, 'w')
            f_test = open(TEST_DATA_BI, 'w')
        output_train = csv.writer(f_train)
        output_train.writerow(['words', 'tags', 'length'])
        output_dev = csv.writer(f_dev)
        output_dev.writerow(['words', 'tags', 'length'])
        output_test = csv.writer(f_test)
        output_test.writerow(['words', 'tags', 'length'])
        self.process_file(TRAIN_PATH, output_train, bigram)
        self.process_file(DEV_PATH, output_dev, bigram)
        self.process_file(TEST_PATH, output_test, bigram)

    def process_data(self, path):
        src_data = []
        data = []
        label = []

        src_data_sentence = []
        data_sentence = []
        label_sentence = []

        f = open(path, 'r')
        li = f.readlines()
        f.close()

        for line in li:
            line = unicode(line, 'utf-8')
            line_t = line.replace('\n', '').replace('\r', '').replace('  ', '#').split('#')
            if len(line_t) < 3:
                if len(data_sentence) == 0:
                    continue
                src_data.append(src_data_sentence)
                data.append(data_sentence)
                label.append(label_sentence)
                src_data_sentence = []
                data_sentence = []
                label_sentence = []
                continue
            src_word = line_t[0]
            word = line_t[1]
            src_data_sentence.append(src_word)
            data_sentence.append(word)
            label_sentence += [line_t[2].split('_')[0]]

        return src_data, data, label
