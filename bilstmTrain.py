import sys
import time
import random
import dynet as dy
import numpy as np
import utils as ut


# globals of the model
EMBED_DIM = 128
HIDDEN_DIM = 50
EPOCHS = 50


class BiLSTMTagger(object):
    """
    2 layer BiLstm tagger model followed by linear/output layer
    """
    def __init__(self):
        self.model = dy.Model()  # generate nn model
        self.trainer = dy.AdamTrainer(self.model)  # trainer of the model
        # embedding layer of the model
        self.embeds = self.model.add_lookup_parameters((len(ut.vocab_set), EMBED_DIM))
        self.builders = [
                            dy.LSTMBuilder(1, EMBED_DIM, HIDDEN_DIM, self.model),
                            dy.LSTMBuilder(1, EMBED_DIM, HIDDEN_DIM, self.model),
                        ]
        self.W = self.model.add_parameters((len(ut.tags_set), HIDDEN_DIM * 2))  # output/linear layer

    def __call__(self, sentence, tags):
        dy.renew_cg()  # new computation graph
        # reset the hidden states of the BiLstm tagger
        f_init, b_init = [b.initial_state() for b in self.builders]
        W = self.W.expr()  # convert the parameter into an expression (add it to graph)
        embeds = [self.embeds[w] for w in sentence]
        fw = [x.output() for x in f_init.add_inputs(embeds)]
        bw = [x.output() for x in b_init.add_inputs(reversed(embeds))]
        losses = []
        for f, b, t in zip(fw, reversed(bw), tags):
            f_b = dy.concatenate([f, b])
            r_t = W * f_b
            err = dy.pickneglogsoftmax(r_t, t)
            losses.append(err)
        return dy.esum(losses)


def predict(sentence, tagger):
    dy.renew_cg()  # new computation graph
    # reset the hidden states of the BiLstm tagger
    f_init, b_init = [b.initial_state() for b in tagger.builders]
    W = tagger.W.expr()  # convert the parameter into an expression (add it to graph)
    embeds = [tagger.embeds[ut.w2i.get(w, ut.w2i[ut.UNK])] for w, t in sentence]
    fw = [x.output() for x in f_init.add_inputs(embeds)]
    bw = [x.output() for x in b_init.add_inputs(reversed(embeds))]
    tags = []
    for f, b in zip(fw, reversed(bw)):
        f_b = dy.concatenate([f, b])
        r_t = W * f_b
        out = dy.softmax(r_t)
        chosen = np.argmax(out.npvalue())
        tags.append(ut.i2t[chosen])
    return tags


def train(train_sentences, dev_sentences, tagger):
    start = time.time()
    sum_of_losses = 0.0
    tagged = 0.0
    for epoch in range(EPOCHS):
        random.shuffle(train_sentences)
        for i, s in enumerate(train_sentences, 1):
            if i % 5000 == 0:
                print('average loss: {}'.format(sum_of_losses / tagged))
                sum_of_losses = 0
                tagged = 0
            if i % 10000 == 0:
                good = bad = 0.0
                for sentence in dev_sentences:
                    tags = predict(sentence, tagger)
                    golds = [t for w, t in sentence]
                    for go, gu in zip(golds, tags):
                        if go == gu:
                            good += 1
                        else:
                            bad += 1
                print('dev results = accuracy: {}%'.format(good / (good + bad)))
            ws = [ut.w2i[w] for w, t in s]
            ts = [ut.t2i[t] for w, t in s]
            sum_errs = tagger(ws, ts)
            sum_of_losses += sum_errs.scalar_value()
            tagged += len(ts)
            sum_errs.backward()  # computing the gradients of the model(backpropagation)
            tagger.trainer.update()  # training step which updating the weights of the model
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
