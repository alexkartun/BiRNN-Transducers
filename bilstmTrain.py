import sys
import time
import random
import dynet as dy
import numpy as np
import utils as ut

# globals of the model
EPOCHS = 50


def train(train_sentences, dev_sentences, tagger):
    start = time.time()

    for epoch in range(EPOCHS):
        random.shuffle(train_sentences)
        for i, s in enumerate(train_sentences, 1):
            tagger.trainer.update()
    end = time.time()
    print('total time of training: {}'.format(end - start))


def main(argv):
    time.sleep(1)  # wait 1 sec for dy packages to be allocated and loaded to memory
    train_file_path = argv[0]
    dev_file_name = argv[1]
    print('generating time...')
    train_sentences = ut.generate_input_sentences(train_file_path)
    dev_sentences = ut.generate_input_sentences(dev_file_name)
    ut.generate_sets_and_dicts(train_sentences)
    print('creating the model...')
    tagger = BiLSTMTagger()
    print('training time...')
    train(train_sentences, dev_sentences, tagger)


if __name__ == '__main__':
    main(sys.argv[1:])
