import dynet as dy
import utils as ut
import numpy as np
# globals of the model
EMBED_DIM = 128
HIDDEN_DIM = 50
HIDDEN_MLP_DIM = 50


class FirstModel(object):
    def __init__(self, w2i, t2i, i2t):
        self.w2i = w2i
        self.t2i = t2i
        self.i2t = i2t
        self.model = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.model)
        self.word_embeddings = self.model.add_lookup_parameters((len(self.w2i), EMBED_DIM))
        self.first_builder = [
            dy.LSTMBuilder(1, EMBED_DIM, HIDDEN_DIM, self.model),
            dy.LSTMBuilder(1, EMBED_DIM, HIDDEN_DIM, self.model),
        ]
        self.second_builder = [
            dy.LSTMBuilder(1, EMBED_DIM, HIDDEN_DIM, self.model),
            dy.LSTMBuilder(1, EMBED_DIM, HIDDEN_DIM, self.model),
        ]
        self.W = self.model.add_parameters((len(self.t2i), HIDDEN_DIM))

    def build_graph(self, sentence):
        dy.renew_cg()
        first_forward, first_backward = [b.initial_state() for b in self.first_builder]
        second_forward, second_backward = [b.initial_state() for b in self.second_builder]
        embeddings = [self.word_repr(word) for word in sentence]
        first_forward_output = first_forward.transduce(embeddings)
        first_backward_output = first_backward.transduce(reversed(embeddings))
        b = [dy.concatenate([fo, bo]) for fo, bo in zip(first_forward_output, first_backward_output)]
        second_forward_output = second_forward.transduce(b)
        second_backward_output = second_backward.transduce(reversed(b))
        b_tag = [dy.concatenate([fo, bo]) for fo, bo in zip(second_forward_output, second_backward_output)]
        W = dy.parameter(self.W)
        result = [W*item for item in b_tag]
        return result

    def word_repr(self, word):
        if word in self.w2i:
            return self.word_embeddings[self.w2i[word]]
        return self.word_embeddings[self.w2i[ut.UNK]]

    def compute_sentence_loss(self, sentence, tags):
        losses = list()
        result = self.build_graph(sentence)
        for res, tag in zip(result, tags):
            loss = dy.pickneglogsoftmax(res, tag)
            losses.append(loss)
        return dy.esum(losses)

    def predict_sentence_tags(self, sentence):
        tags = list()
        result = self.build_graph(sentence)
        for res in result:
            out = dy.softmax(res)
            chosen = np.argmax(out.npvalue())
            tags.append(self.i2t[chosen])
        return tags


class SecondModel(FirstModel):
    def __init__(self, w2i, t2i, i2t, c2i):
        super(SecondModel, self).__init__(w2i, t2i, i2t)
        self.c2i = c2i
        self.char_embeddings = self.model.add_lookup_parameters((len(self.c2i), EMBED_DIM))
        self.builder = dy.LSTMBuilder(1, EMBED_DIM, HIDDEN_DIM, self.model)

    def word_repr(self, word):
        char_indexes = [self.c2i[c] for c in word]
        char_embeddings = [self.char_embeddings[i] for i in char_indexes]
        lstm = self.builder.initial_state()
        output = lstm.transduce(char_embeddings)
        return output[-1]


class ThirdModel(FirstModel):
    def __init__(self, w2i, t2i, i2t, p2i, s2i, i2s):
        super(ThirdModel, self).__init__(w2i, t2i, i2t)
        self.p2i = p2i
        self.s2i = s2i
        self.i2s = i2s
        self.prefix_embeddings = self.model.add_lookup_parameters(len(self.p2i), EMBED_DIM)
        self.suffix_embeddings = self.model.add_lookup_parameters(len(self.s2i), EMBED_DIM)

    def word_repr(self, word):
        prefix = word[:ut.SUB_WORD_UNIT_SIZE]
        suffix = word[-ut.SUB_WORD_UNIT_SIZE:]
        if prefix in self.p2i:
            prefix_embedding = self.prefix_embeddings[self.p2i[prefix]]
        else:
            prefix_embedding = self.prefix_embeddings[self.p2i[ut.UNK[:ut.SUB_WORD_UNIT_SIZE]]]
        if suffix in self.p2i:
            suffix_embedding = self.suffix_embeddings[self.p2i[suffix]]
        else:
            suffix_embedding = self.suffix_embeddings[self.p2i[ut.UNK[-ut.SUB_WORD_UNIT_SIZE:]]]
        return dy.esum([prefix_embedding, suffix_embedding])


class FourthModel(SecondModel):
    def __init__(self, w2i, t2i, i2t, c2i):
        super(FourthModel, self).__init__(w2i, t2i, i2t, c2i)
        self.W = self.model.add_parameters((EMBED_DIM, HIDDEN_DIM * 2))

    def word_repr(self, word):
        word_embeddings = FirstModel.word_repr(self, word)
        character_embeddings = SecondModel.word_repr(self, word)
        concat_embeddings = dy.concatenate([word_embeddings, character_embeddings])
        W = dy.parameter(self.W)
        result = W*concat_embeddings
        return result
