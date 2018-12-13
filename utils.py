# util sets and dicts
UNK = '_UNK_'
SUB_WORD_UNIT_SIZE = 3
vocab_set = {UNK}
tags_set = set()
w2i = dict()
i2w = dict()
t2i = dict()
i2t = dict()
c2i = dict()
p2i = dict()
s2i = dict()


def generate_validation_data(filename):
    """
    generating list sentences where each sentence combined of examples/words to be tagged
    :param filename: file name of test data set
    :return: list of sentences
    """
    sentences = list()
    sentence = list()
    with open(filename, 'r') as f:
        for line in f.readlines():
            if not line.strip():
                sentences.append(sentence)
                sentence = list()
                continue
            word = line.strip()
            sentence.append(word)
    return sentences


def generate_input_data(filename):
    """
    generating list of tuples where each tuple is (sentence, tags_of_the_sentence)
    :param filename: file name of data set
    :return: list of tuples(train or dev data)
    """
    data = list()
    sentence = list()
    tags = list()
    with open(filename, 'r') as f:
        for line in f.readlines():
            if not line.strip():
                data.append((sentence, tags))
                sentence = list()
                tags = list()
                continue
            word, tag = line.strip().split()
            sentence.append(word)
            tags.append(tag)
    return data


def generate_sets_and_dicts(train_data):
    """
    generating vocab set and tag set from the train data,
    generating the util dictionaries for fast lookup of the the words and tags to their specific index,
    and vise versa
    :param train_data: train data list
    :return:
    """
    global w2i, i2w, t2i, i2t, c2i, p2i, s2i
    for sentence, tags in train_data:
        for word, tag in zip(sentence, tags):
            vocab_set.add(word)
            tags_set.add(tag)

    w2i = {w: i for i, w in enumerate(vocab_set)}
    i2w = {i: w for w, i in w2i.items()}
    t2i = {c: i for i, c in enumerate(tags_set)}
    i2t = {i: c for c, i in t2i.items()}
    c2i = {c: i for word in vocab_set for i, c in enumerate(word)}
    p2i = {w[:SUB_WORD_UNIT_SIZE]: i for i, w in enumerate(vocab_set)}
    s2i = {w[-SUB_WORD_UNIT_SIZE:]: i for i, w in enumerate(vocab_set)}