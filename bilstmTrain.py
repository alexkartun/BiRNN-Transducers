import sys
import time
import random
import utils as ut
import bilstmModels as bm

STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}

# globals of the model
EPOCHS = 5
NUMBER_SENTENCES_CHECK_ACCURACY = 500


def train(train_data, dev_data, model):
    """
    training the model and checking accuracy on dev every 500 sentences
    :param train_data: training data
    :param dev_data: dev data
    :param model: BiLSTM 2 layer model with specific word embedding representation
    :return:
    """
    start = time.time()
    for epoch in range(EPOCHS):
        sum_of_losses = 0.0
        tagged_words = 0.0
        random.shuffle(train_data)
        for index, (sentence, tags) in enumerate(train_data, 1):
            if index % NUMBER_SENTENCES_CHECK_ACCURACY == 0:
                print('dev results = accuracy: {}%'.format(compute_accuracy(dev_data, model)))
            # computing model's errors on the sentence
            sum_losses = model.compute_loss(sentence, tags)
            sum_of_losses += sum_losses.value()
            tagged_words += len(tags)
            sum_losses.backward()
            model.trainer.update()
        print('train results = epoch: {}, average loss: {}'.format(epoch, sum_of_losses / tagged_words))
    end = time.time()
    print('total time of training: {}'.format(end - start))


def compute_accuracy(data, model):
    """
    computing accuracy of the model on data
    :param data: data to be checked
    :param model: trained model
    :return:
    """
    good = 0.0
    bad = 0.0
    for sentence, sentence_gold_tags in data:
        # computing model's predictions on the sentence
        sentence_predicted_tags = model.compute_prediction(sentence)
        for predicted_tag, gold_tag in zip(sentence_predicted_tags, sentence_gold_tags):
            if predicted_tag == gold_tag:
                good += 1
            else:
                bad += 1
    return 100 * (good / (good + bad))


def create_model(word_representation):
    """
    creating specific model which depends on word_representation as user's input
    :param word_representation: user's input for word representation
    :return: specific model
    """
    if word_representation == 'a':
        return bm.FirstModel(ut.w2i, ut.t2i, ut.i2t)
    elif word_representation == 'b':
        return bm.SecondModel(ut.w2i, ut.t2i, ut.i2t, ut.c2i)
    elif word_representation == 'c':
        return bm.ThirdModel(ut.w2i, ut.t2i, ut.i2t, ut.p2i, ut.s2i)
    else:
        return bm.FourthModel(ut.w2i, ut.t2i, ut.i2t, ut.c2i)


def main(argv):
    time.sleep(1)  # wait 1 sec for dy packages to be allocated and loaded to memory
    word_representation = argv[0]
    train_file_path = argv[1]
    dev_file_path = argv[2]
    # model_file_path = argv[3]
    print('generating the data...')
    train_data = ut.generate_input_data(train_file_path)
    ut.generate_sets_and_dicts(train_data)
    dev_data = ut.generate_input_data(dev_file_path)
    print('creating the model...')
    model = create_model(word_representation)
    print('training...')
    train(train_data, dev_data, model)


if __name__ == '__main__':
    main(sys.argv[1:])
