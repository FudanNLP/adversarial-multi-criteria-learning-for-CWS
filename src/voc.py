import numpy as np
from collections import defaultdict


class Vocab(object):
    def __init__(self, path_vec, train_word, single_task, bi_gram, frequency=15):
        self.path = path_vec
        self.table_path = train_word
        self.word2idx = defaultdict(int)
        self.word_vectors = None
        self.single = single_task
        self.bigram = bi_gram
        self.frequency = frequency
        self.table = []
        self.process_table(self.table_path, self.single)
        self.load_data()

    def process_table(self, word_path, single):
        if self.bigram:
            f = open(word_path, 'r')
            text = f.readlines()
            if single is False:
                for line in text:
                    com = unicode(line, 'utf-8').strip()
                    self.table.append(com)
            else:
                table = []
                for line in text:
                    com = unicode(line, 'utf-8').strip().split(' ')
                    if len(com[1]) == 2 and int(com[2]) > self.frequency :
                        table.append(com[1])
                self.table = set(table)
            f.close()
        else:
            self.table = set()

    def load_data(self):
        with open(self.path, 'r') as f:
            line = f.readline().strip().split(" ")
            N, dim = map(int, line)

            self.word_vectors = []
            idx = 0
            for k in range(N):
                line = unicode(f.readline(), 'utf-8').strip().split(" ")
                self.word2idx[line[0]] = idx
                vector = np.asarray(map(float, line[1:]), dtype=np.float32)
                self.word_vectors.append(vector)
                idx += 1

            count = 0
            for word in self.table:
                mean_vec = np.zeros(dim)
                for ch in word:
                    if ch in self.word2idx:
                        mean_vec += self.word_vectors[self.word2idx[ch]]
                        count += 1
                    else:
                        mean_vec += self.word_vectors[self.word2idx['<OOV>']]
                word_vec = mean_vec / 2.0
                self.word2idx[word] = idx
                self.word_vectors.append(word_vec)
                idx += 1

            print 'Vocab size:', len(self.word_vectors)
            print 'word2idx:', len(self.word2idx)
            print 'index:', idx
            print 'count:', count

            self.word_vectors = np.asarray(self.word_vectors, dtype=np.float32)

class Tag(object):
    def __init__(self):
        self.tag2idx = defaultdict(int)
        self.define_tags()

    def define_tags(self):
        self.tag2idx['B'] = 0
        self.tag2idx['M'] = 1
        self.tag2idx['E'] = 2
        self.tag2idx['S'] = 3

class OOV(object):
    def __init__(self, dictpath, goldpath, testpath, destpath, yp):
        self.path = dictpath
        self.goldpath = goldpath
        self.testpath = testpath
        self.destpath = destpath
        self.dict = defaultdict(int)
        self.word_dict()
        self.ans_segs = self.prod_ans()
        self.pred_segs = self.prod_pred(self.process_data(), yp)


    def word_dict(self):
        f = open(self.path, 'r')
        li = f.readlines()
        f.close()
        for line in li:
            line = line.strip().decode('utf-8')
            self.dict[line] = 1

    def prod_ans(self):
        f = open(self.goldpath, 'r')
        li = f.readlines()
        f.close()
        ans_segs = []
        for line in li:
            line = line.strip().decode('utf-8').split(' ')
            sent = []
            for word in line:
                sent.append(word)
            ans_segs.append(sent)
        return ans_segs

    def process_data(self):
        src_data = []

        src_data_sentence = []

        f = open(self.testpath, 'r')
        li = f.readlines()
        f.close()

        for line in li:
            line = unicode(line, 'utf-8')
            line_t = line.replace('\n', '').replace('\r', '').replace('  ', '#').split('#')
            if (len(line_t) < 3):
                if (len(src_data_sentence) == 0):
                    continue
                src_data.append(src_data_sentence)
                src_data_sentence = []
                continue
            src_word = line_t[0]
            # print src_word,word
            src_data_sentence.append(src_word)

        return src_data

    def prod_pred(self, src_data, seq):
        newSeq = []
        start = 0
        pred_seg = []
        f = open(self.destpath, 'w')
        for line in src_data:
            length = len(line)
            end = start + length
            newSeq.append(seq[start:end])
            start = end

        pred_sent = []
        words = ''
        for line, tags in zip(src_data, newSeq):
            for word, label in zip(line, tags):
                word = word.encode('utf-8')
                words += word
                f.write(word)
                if (label == 2 or label == 3):
                    f.write(' ')
                    pred_sent.append(words.decode('utf-8'))
                    words = ''
            pred_seg.append(pred_sent)
            pred_sent = []
            f.write('\n')

        f.close()
        return pred_seg

    def eval_oov_rate(self):
        right = 0
        total = 0
        for ans_sentence, pred_sentence in zip(self.ans_segs, self.pred_segs):
            ans = []
            for word in ans_sentence:
                ans.append(word)
                for i in xrange(len(word) - 1): ans.append(-1)
            pred = []
            for word in pred_sentence:
                pred.append(word)
                for i in xrange(len(word) - 1): pred.append(-1)
            for ans_word, pred_word in zip(ans, pred):
                if ans_word == -1: continue
                if self.dict.get(ans_word) == None:
                    total += 1
                    if pred_word == -1: continue
                    if len(ans_word) != len(pred_word): continue
                    if ans_word.find(pred_word) == -1: continue
                    right += 1
        oov_recall_rate = right * 1.0 / total
        print ('total=', total, 'right=', right, 'oov_recall_rate=',oov_recall_rate)
        return oov_recall_rate


