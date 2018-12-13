import sys
import time
import random
import dynet as dy
import numpy as np

# util sets and dicts
vocab_set = set()
tags_set = set()
c2i = dict()
i2c = dict()
t2i = dict()
i2t = dict()

# globals of the model
EMBED_DIM = 128
HIDDEN_DIM = 50
HIDDEN_MLP_DIM = 50
EPOCHS = 2


def generate_data(filename):
    """
    generating list of examples, where each example is tuple in format of (word, tag)
    :param filename: file name of data set
    :return: list of examples
    """
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            word, tag = line.strip().split()
            data.append((word, tag))
    return data


def generate_sets_and_dicts(train_data):
    """
    generating vocabulary set and tag set from the train data,
    generating the util dictionaries for fast lookup of the the characters and tags to their specific index,
    and vise versa
    :param train_data: train data set
    :return:
    """
    global c2i, i2c, t2i, i2t
    for word, tag in train_data:
        for c in word:
            vocab_set.add(c)
        tags_set.add(tag)
    c2i = {c: i for i, c in enumerate(vocab_set)}
    i2c = {i: c for c, i in c2i.items()}
    t2i = {c: i for i, c in enumerate(tags_set)}
    i2t = {i: c for c, i in t2i.items()}


class LSTMAcceptor(object):
    """
    1 layer Lstm acceptor model followed by MLP with one hidden layer
    """
    def __init__(self):
        self.model = dy.ParameterCollection()  # generate nn model
        self.trainer = dy.AdamTrainer(self.model)  # trainer of the model
        self.char_embeddings = self.model.add_lookup_parameters((len(c2i), EMBED_DIM))  # embedding layer of the model
        self.builder = dy.LSTMBuilder(1, EMBED_DIM, HIDDEN_DIM, self.model)  # lstm layer
        self.W1 = self.model.add_parameters((HIDDEN_MLP_DIM, HIDDEN_DIM))  # hidden layer of the mlp
        self.W2 = self.model.add_parameters((len(t2i), HIDDEN_MLP_DIM))  # output layer of the mlp

    def build_graph(self, word):
        """
        building new computation graph with previous parameters and computing model's prediction on the word
        :param word: word to be predicted by the model
        :return: predicted result
        """
        dy.renew_cg()  # new computation graph
        lstm = self.builder.initial_state()
        char_embeddings = self.word_repr(word)
        outputs = lstm.transduce(char_embeddings)
        W1 = dy.parameter(self.W1)
        W2 = dy.parameter(self.W2)
        result = W2 * (dy.tanh(W1 * outputs[-1]))
        return result

    def word_repr(self, word):
        """
        computing character-level embedding representation of the word
        :param word: word to be embedded
        :return: word embedding representation
        """
        char_indexes = [c2i[c] for c in word]
        char_embeddings = [self.char_embeddings[i] for i in char_indexes]
        return char_embeddings

    def compute_word_loss(self, word, tag):
        """
        computing model's prediction and computing negative cross entropy loss of the predicted value and gold value
        :param word: word to be predicted
        :param tag: gold label
        :return: error of the model on his predicted label
        """
        result = self.build_graph(word)
        loss = dy.pickneglogsoftmax(result, t2i[tag])
        return loss

    def predict_word_tag(self, word):
        """
        predicting the label of the word by the computation graph, distributing the values by softmax
        and finally taking the highest valued index as chosen predicted label
        :param word: word to be predicted
        :return: chosen label of the model
        """
        result = self.build_graph(word)
        out = dy.softmax(result)
        chosen = i2t[np.argmax(out.npvalue())]
        return chosen


def train(train_data, dev_data, acceptor):
    """
    training the acceptor lstm model by training data set,
    printing the average loss of the model per each epoch,
    each epoch printing the average loss and accuracy of the model on dev data set
    :param train_data: train data set
    :param dev_data: dev data set
    :param acceptor: acceptor lstm model
    :return:
    """
    start = time.time()
    for epoch in range(EPOCHS):
        sum_of_losses = 0.0
        random.shuffle(train_data)
        for word, tag in train_data:
            # computing loss of the model on this word and gold tag
            loss = acceptor.compute_word_loss(word, tag)
            sum_of_losses += loss.npvalue()  # summing this loss to overall loss
            loss.backward()  # computing the gradients of the model(backpropagation)
            acceptor.trainer.update()  # training step which updating the weights of the model
        print('train results = epoch: {}, accuracy: {}%, average loss: {}'.format(epoch, compute_accuracy(train_data,
                                                                                                          acceptor),
                                                                                  sum_of_losses / len(train_data)))
        evaluate(dev_data, acceptor)
    end = time.time()
    print('total time of training: {}'.format(end - start))


def evaluate(dev_data, acceptor):
    """
    evaluating the accuracy and loss of each example in the dev data set
    printing the accuracy and the average loss of the model on the dev data set
    :param dev_data: dev data set
    :param acceptor: acceptor lstm model
    :return:
    """
    sum_of_losses = 0.0
    for word, tag in dev_data:
        loss = acceptor.compute_word_loss(word, tag)  # computing loss of the model on this word and gold tag
        sum_of_losses += loss.npvalue()  # summing this loss to overall loss
        loss.backward()  # computing the gradients of the model(backpropagation)
        acceptor.trainer.update()  # training step which updating the weights of the model
    print('dev results = accuracy: {}%, average loss: {}'.format(compute_accuracy(dev_data, acceptor),
                                                                 sum_of_losses / len(dev_data)))


def compute_accuracy(data, acceptor):
    """
    computing the accuracy of the model's predictions on the data
    :param data: data to be predicted
    :param acceptor: acceptor lstm model
    :return: computed accuracy
    """
    correct = 0
    for word, tag in data:
        predicted_label = acceptor.predict_word_tag(word)
        if predicted_label == tag:
            correct += 1
    return 100 * (correct / len(data))


def main(argv):
    time.sleep(1)  # wait 1 sec for dy packages to be allocated and loaded to memory
    train_file_path = argv[0]
    dev_file_path = argv[1]
    print('generating time...')
    train_data = generate_data(train_file_path)
    dev_data = generate_data(dev_file_path)
    generate_sets_and_dicts(train_data)
    print('creating the model...')
    acceptor = LSTMAcceptor()
    print('training time...')
    train(train_data, dev_data, acceptor)


if __name__ == '__main__':
    main(sys.argv[1:])
