import dynet as dy
import numpy as np
import utils as ut
STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}

# model globals
WORD_EMBED_DIM = 100
LSTM_LAYER_SIZE = 1
HIDDEN_DIM = 50
CHAR_EMBEDDING_DIM = 20
SUBWORD_EMBEDDING_DIM = 100


class BiLSTM(object):
    """
    1 layer BiLSTM
    """
    def __init__(self, in_dim, model):
        self.builders = [
            dy.LSTMBuilder(LSTM_LAYER_SIZE, in_dim, HIDDEN_DIM, model),
            dy.LSTMBuilder(LSTM_LAYER_SIZE, in_dim, HIDDEN_DIM, model)
        ]

    def __call__(self, sentence):
        forward_init, backward_init = [b.initial_state() for b in self.builders]
        forward_output = forward_init.transduce(sentence)
        backward_output = backward_init.transduce(reversed(sentence))
        return [dy.concatenate([fo, bo]) for fo, bo in zip(forward_output, backward_output)]


class FirstModel(object):
    """
    First model as main model that contains:
    2 layers BiLSTM with output to mlp with 1 output layer with 'softmax' as distribution function.
    """
    def __init__(self, w2i, t2i, i2t):
        self.w2i = w2i
        self.t2i = t2i
        self.i2t = i2t
        self.model = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.model)
        # word embedding layer
        self.E = self.model.add_lookup_parameters((len(self.w2i), WORD_EMBED_DIM))
        self.firstBiLSTM = BiLSTM(WORD_EMBED_DIM, self.model)
        self.secondBiLSTM = BiLSTM(2 * HIDDEN_DIM, self.model)
        # output mlp layer
        self.W = self.model.add_parameters((len(self.t2i), 2 * HIDDEN_DIM))
        self.b = self.model.add_parameters(len(self.t2i))

    def __call__(self, sentence):
        dy.renew_cg()  # create graph again
        embeddings = [self.represent(word) for word in sentence]
        firstBiLSTM_output = self.firstBiLSTM(embeddings)
        secondBiLSTM_output = self.secondBiLSTM(firstBiLSTM_output)
        return [dy.softmax(self.W * output + self.b) for output in secondBiLSTM_output]

    def represent(self, word):
        """
        word's embedding representation for existing word otherwise 'UNK" embedding representation
        :param word: word to represent
        :return: word's embedding representation
        """
        if word in self.w2i:
            return self.E[self.w2i[word]]
        return self.E[self.w2i[ut.UNK]]

    def compute_loss(self, sentence, tags):
        """
        computing loss of every word in sentence
        :param sentence: sentence words to be predicted
        :param tags: golden tags of the sentence
        :return: sum of all losses
        """
        sentence_losses = []
        sentence_predicted_tags = self(sentence)
        for word_predicted_tag, gold_tag in zip(sentence_predicted_tags, tags):
            # loss calculated -log(pred_tag, golden_tag)
            sentence_losses.append(-dy.log(dy.pick(word_predicted_tag, self.t2i[gold_tag])))
        return dy.esum(sentence_losses)

    def compute_prediction(self, sentence):
        """
        computing prediction tag of every word in sentence
        :param sentence: sentence of words to be predicted
        :return: predicted tags of the sentence
        """
        sentence_tags = []
        sentence_predicted_tags = self(sentence)
        for word_predicted_tag in sentence_predicted_tags:
            sentence_tags.append(self.i2t[np.argmax(word_predicted_tag.value())])
        return sentence_tags


class SecondModel(FirstModel):
    """
    Second model with char embedding representation that passed to 1 layer lstm
    """
    def __init__(self, w2i, t2i, i2t, c2i):
        FirstModel.__init__(self, w2i, t2i, i2t)
        self.c2i = c2i
        self.char_embeddings = self.model.add_lookup_parameters((len(self.c2i), CHAR_EMBEDDING_DIM))
        self.LSTMc = dy.LSTMBuilder(LSTM_LAYER_SIZE, CHAR_EMBEDDING_DIM, WORD_EMBED_DIM, self.model)

    def represent(self, word):
        LSTMc_init_state = self.LSTMc.initial_state()
        char_embeddings = [self.char_embeddings[self.c2i[char]] for char in word]
        return LSTMc_init_state.transduce(char_embeddings)[-1]


class ThirdModel(FirstModel):
    """
    Third model which representation is summing of suffix, prefix and word embeddings
    """
    def __init__(self, w2i, t2i, i2t, p2i, s2i):
        FirstModel.__init__(self, w2i, t2i, i2t)
        self.p2i = p2i
        self.s2i = s2i
        self.prefix_embeddings = self.model.add_lookup_parameters((len(self.p2i), SUBWORD_EMBEDDING_DIM))
        self.suffix_embeddings = self.model.add_lookup_parameters((len(self.s2i), SUBWORD_EMBEDDING_DIM))

    def represent(self, word):
        word_embedding = FirstModel.represent(self, word)
        prefix_embedding = self.prefix_embeddings[self.p2i[self.compute_word_prefix(word)]]
        suffix_embedding = self.suffix_embeddings[self.s2i[self.compute_word_suffix(word)]]
        return word_embedding + prefix_embedding + suffix_embedding

    def compute_word_prefix(self, word):
        """
        computing word prefix of word if exists otherwise prefix of 'UNK'
        :param word: word to be checked
        :return: prefix of the word
        """
        if word in self.w2i:
            return word[:ut.SUB_WORD_UNIT_SIZE]
        return ut.UNK[:ut.SUB_WORD_UNIT_SIZE]

    def compute_word_suffix(self, word):
        """
        computing word suffix of word if exists otherwise suffix of 'UNK'
        :param word: word to be checked
        :return: suffix of the word
        """
        if word in self.w2i:
            return word[-ut.SUB_WORD_UNIT_SIZE:]
        return ut.UNK[-ut.SUB_WORD_UNIT_SIZE:]


class FourthModel(SecondModel):
    """
    Fourth model which representation is concating of word and char embeddings, passed to 1 output linear layer
    """
    def __init__(self, w2i, t2i, i2t, c2i):
        SecondModel.__init__(self, w2i, t2i, i2t, c2i)
        self.Linear = self.model.add_parameters((WORD_EMBED_DIM, 2 * WORD_EMBED_DIM))
        self.bias = self.model.add_parameters(WORD_EMBED_DIM)

    def represent(self, word):
        word_embeddings = FirstModel.represent(self, word)
        char_embeddings = SecondModel.represent(self, word)
        return self.Linear * dy.concatenate([word_embeddings, char_embeddings]) + self.bias
