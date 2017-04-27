import os

def process_data(output):
    fw = open(output, 'w')
    words = []

    for dirpath, dirnames, filenames in os.walk(os.curdir):
        for filename in filenames:
            if filename.endswith('words_for_training'):
                path = os.path.join(dirpath, filename)
                print path
                f = open(path, 'r')
                li = f.readlines()
                for line in li:
                    com = unicode(line, 'utf-8').strip().split(' ')
                    if len(com[1]) == 2 and int(com[2]) > 15:
                        words.append(com[1])
                f.close()

    word_dict = set(words)
    print len(words)
    print len(word_dict)

    for ele in word_dict:
        fw.write(ele.encode('utf-8') + '\n')

    fw.close()

process_data('models/train_words')

