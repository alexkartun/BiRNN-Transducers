# util sets and dicts
UNK = '_UNK_'
vocab_set = {UNK}
tags_set = set()
w2i = dict()
i2w = dict()
t2i = dict()
i2t = dict()


def generate_validation_sentences(filename):
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


def generate_input_sentences(filename):
    """
    generating list of sentences where each sentence combined of tuple (word, tag)
    :param filename: file name of data set
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
            word, tag = line.strip().split()
            sentence.append((word, tag))
    return sentences


def generate_sets_and_dicts(train_sentences):
    """
    generating character set and tag set from the train data,
    generating the util dictionaries for fast lookup of the the words and tags to their specific index,
    and vise versa
    :param train_sentences: train data set sentences
    :return:
    """
    global w2i, i2w, t2i, i2t
    for sentence in train_sentences:
        for word, tag in sentence:
            vocab_set.add(word)
            tags_set.add(tag)

    w2i = {c: i for i, c in enumerate(vocab_set)}
    i2w = {i: c for c, i in w2i.items()}
    t2i = {c: i for i, c in enumerate(tags_set)}
    i2t = {i: c for c, i in t2i.items()}
