import sys
import time
import dynet as dy
import numpy as np

# util sets and dicts
characters_set = set()
tags_set = set()
c2i = dict()
i2c = dict()
t2i = dict()
i2t = dict()

# globals of the model
EMBED_DIM = 100
HIDDEN_DIM = 100
HIDDEN_MLP_DIM = 100
EPOCHS = 3


def generate_data(filename):
    """
    generating from the data list of tuples, where each tuple is in form of (word, tag)
    :param filename: file's name
    :return: list of tuples
    """
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            word, tag = line.strip().split()
            data.append((word, tag))
    return data


def generate_sets(train_data):
    """
    generating character set and tag set from the train data,
    generating the util dictionaries for fast lookup of the the characters and tags to their specific index,
    and vise versa
    :param train_data:
    :return:
    """
    global c2i, i2c, t2i, i2t
    for word, tag in train_data:
        for c in word:
            characters_set.add(c)
        tags_set.add(tag)
    c2i = {c: i for i, c in enumerate(characters_set)}
    i2c = {i: c for c, i in c2i.items()}
    t2i = {c: i for i, c in enumerate(tags_set)}
    i2t = {i: c for c, i in t2i.items()}


class LstmAcceptor(object):
    """
    Lstm acceptor model followed by MLP with one hidden layer
    """
    def __init__(self):
        self.model = dy.Model()  # generate nn model
        self.trainer = dy.AdamTrainer(self.model)  # trainer of the model
        # embedding layer of the model
        self.embeds = self.model.add_lookup_parameters((len(characters_set), EMBED_DIM))
        self.builder = dy.VanillaLSTMBuilder(1, EMBED_DIM, HIDDEN_DIM, self.model)  # lstm layer
        self.W1 = self.model.add_parameters((HIDDEN_MLP_DIM, HIDDEN_DIM))  # hidden layer of the mlp
        self.W2 = self.model.add_parameters((len(tags_set), HIDDEN_MLP_DIM))  # output layer of the mlp

    def __call__(self, sequence):
        lstm = self.builder.initial_state()  # reset the hidden states of the lstm acceptor
        W1 = self.W1.expr()  # convert the parameter into an expression (add it to graph)
        W2 = self.W2.expr()  # convert the parameter into an expression (add it to graph)
        # first calculate the embedded vectors of the sequence and second send the vectors to the lstm acceptor
        outputs = lstm.transduce([self.embeds[c2i[i]] for i in sequence])
        # last state of the lstm output multiplied with hidden layer of mlp followed by activation 'tanh' function
        outputs = dy.tanh(W1*outputs[-1])
        # hidden layer's output multiplied by output layer of the mlp
        result = W2*outputs
        return result


def train(train_data, test_data, acceptor):
    """
    training the acceptor lstm model by training data set,
    printing the average loss of the model per each epoch
    :param train_data: train data set
    :param test_data: test data set
    :param acceptor: acceptor lstm model
    :return:
    """
    start = time.time()
    for epoch in range(EPOCHS):
        sum_of_losses = 0.0
        for sequence, label in train_data:
            dy.renew_cg()  # new computation graph
            preds = acceptor(sequence)  # compute the predicted expression of the model on the sequence
            # calculate negative cross entropy loss of the predicted expression which distributed by softmax
            loss = dy.pickneglogsoftmax(preds, t2i[label])
            sum_of_losses += loss.npvalue()  # summing this loss to overall loss
            loss.backward()  # computing the gradients of the model(backpropagation)
            acceptor.trainer.update()  # training step which updating the weights of the model
        print('train results = epoch: {}, accuracy: {}%, average loss: {}'.format(epoch, compute_accuracy(train_data,
                                                                                                          acceptor),
                                                                                  sum_of_losses / len(train_data)))
        evaluate(test_data, acceptor)
    end = time.time()
    print('total time of training: {}'.format(end - start))


def evaluate(test_data, acceptor):
    """
    evaluating the accuracy and loss of each example in the test data
    printing the accuracy and the average loss of the model on the test set
    :param test_data: test data set
    :param acceptor: acceptor lstm model
    :return:
    """
    sum_of_losses = 0.0
    for sequence, label in test_data:
        dy.renew_cg()  # new computation graph
        preds = acceptor(sequence)  # compute the predicted expression of the model on the sequence
        # calculate negative cross entropy loss of the predicted expression which distributed by softmax
        loss = dy.pickneglogsoftmax(preds, t2i[label])
        sum_of_losses += loss.npvalue()  # summing this loss to overall loss
        loss.backward()  # computing the gradients of the model(backpropagation)
        acceptor.trainer.update()  # training step which updating the weights of the model
    print('test results = accuracy: {}%, average loss: {}'.format(compute_accuracy(test_data, acceptor),
                                                                  sum_of_losses / len(test_data)))


def compute_accuracy(data, acceptor):
    """
    computing the accuracy of the model's predictions on the data
    :param data: data to be predicted
    :param acceptor: acceptor lstm model
    :return: computed accuracy
    """
    correct = 0
    for sequence, label in data:
        predicted_label = predict(sequence, acceptor)
        if predicted_label == label:
            correct += 1
    return 100 * (correct / len(data))


def predict(sequence, acceptor):
    """
    predicting the label of the sequence
    :param sequence: sequence to be predicted
    :param acceptor: acceptor lstm model
    :return: predicted label
    """
    dy.renew_cg()  # new computation graph
    # computing the predicting expression of the model on the sequence, that distributed by softmax
    preds = dy.softmax(acceptor(sequence))
    # computing the value of preds in computation graph
    vals = preds.npvalue()
    return i2t[np.argmax(vals)]


def main(argv):
    time.sleep(1)  # wait 1 sec for dy packages to be allocated and loaded to memory
    train_file_path = argv[0]
    test_file_path = argv[1]
    print('generating time...')
    train_data = generate_data(train_file_path)
    test_data = generate_data(test_file_path)
    generate_sets(train_data)
    print('creating the model...')
    acceptor = LstmAcceptor()
    print('training time...')
    train(train_data, test_data, acceptor)


if __name__ == '__main__':
    main(sys.argv[1:])
