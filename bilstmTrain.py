import sys
import time
import random
import utils as ut
import bilstmModels as bm

# globals of the model
EPOCHS = 5
NUMBER_SENTENCES_CHECK_ACCURACY = 500


def train(train_data, dev_data, model):
    start = time.time()
    for epoch in range(EPOCHS):
        sum_of_losses = 0.0
        tagged_words = 0.0
        random.shuffle(train_data)
        for index, (sentence, tags) in enumerate(train_data, 1):
            if index % NUMBER_SENTENCES_CHECK_ACCURACY == 0:
                evaluate(dev_data, model)
            sum_losses = model.compute_sentence_loss(sentence, tags)
            sum_of_losses += sum_losses.value()
            tagged_words += len(tags)
            sum_losses.backward()
            model.trainer.update()
        print('train results = epoch: {}, average loss: {}'.format(epoch, sum_of_losses / tagged_words))
    end = time.time()
    print('total time of training: {}'.format(end - start))


def evaluate(dev_data, model):
    sum_of_losses = 0.0
    tagged_words = 0.0
    for sentence, tags in dev_data:
        sum_losses = model.compute_sentence_loss(sentence, tags)
        sum_of_losses += sum_losses.value()
        tagged_words += len(tags)
        sum_losses.backward()
        model.trainer.update()
    print('dev results = accuracy: {}%, average loss: {}'.format(compute_accuracy(dev_data, model),
                                                                 sum_of_losses / tagged_words))


def compute_accuracy(data, model):
    correct = 0
    tagged_words = 0.0
    for sentence, tags in data:
        predicted_tags = model.predict_sentence_tags(sentence)
        for predicted_tag, gold_tag in zip(predicted_tags, tags):
            if predicted_tag == gold_tag:
                correct += 1
            tagged_words += 1
    return 100 * (correct / tagged_words)


def create_model(word_representation):
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
    print('generating time...')
    train_data = ut.generate_input_data(train_file_path)
    dev_data = ut.generate_input_data(dev_file_path)
    ut.generate_sets_and_dicts(train_data)
    print('creating the model...')
    model = create_model(word_representation)
    print('training time...')
    train(train_data, dev_data, model)


if __name__ == '__main__':
    main(sys.argv[1:])
