STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}

# util sets and dicts
UNK = '_UNK_'
SUB_WORD_UNIT_SIZE = 3
vocab_set = {UNK}
tags_set = set()
w2i = {}
i2w = {}
t2i = {}
i2t = {}
c2i = {}
p2i = {}
s2i = {}


def generate_validation_data(filename):
    """
    generating list of sentences where each sentence combined of examples/words to be tagged
    :param filename: file name of test data set
    :return: list of sentences
    """
    data = []
    sentence = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if not line.strip():
                data.append(sentence)
                sentence = []
                continue
            word = line.strip()
            sentence.append(word)
    return data


def generate_input_data(filename):
    """
    generating list of tuples where each tuple is (sentence, tags_of_the_sentence)
    :param filename: file name of data set
    :return: list of tuples(train or dev data)
    """
    data = []
    sentence = []
    tags = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if not line.strip():
                data.append((sentence, tags))
                sentence = []
                tags = []
                continue
            word, tag = line.strip().split()
            sentence.append(word)
            tags.append(tag)
    return data


def generate_sets_and_dicts(train_data):
    """
    generating vocab set and tag set from the train data,
    generating the util dictionaries for fast lookup of the the words and tags to their specific index,
    and vise versa, char to index lookup and prefix & suffix to index lookup
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
    # create prefixes and suffixes
    prefixes = {word[:SUB_WORD_UNIT_SIZE] for word in vocab_set}
    suffixes = {word[-SUB_WORD_UNIT_SIZE:] for word in vocab_set}
    p2i = {w: i for i, w in enumerate(prefixes)}
    s2i = {w: i for i, w in enumerate(suffixes)}
