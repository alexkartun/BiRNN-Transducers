import os
import sys
import time
import utils as ut
import bilstmModels as bm
import pickle
from zipfile import ZipFile

STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}


def predicting(blind_data, model):
    """
    predicting process of already trained model
    :param blind_data: blind data to be tagged
    :param model: trained model for predictions
    :return: all predicted outputs
    """
    start = time.time()
    predicted_outputs = []
    for sentence in blind_data:
        predicted_outputs.append(model.compute_prediction(sentence))
    end = time.time()
    print('total time of training: {}'.format(end - start))
    return predicted_outputs


def create_model(word_representation, dictionaries):
    """
    creating specific model which depends on word_representation as user's input
    :param dictionaries: model's dictionaries
    :param word_representation: user's input for word representation
    :return: specific model
    """
    w2i, t2i, i2t, c2i, p2i, s2i = dictionaries
    if word_representation == 'a':
        return bm.FirstModel(w2i, t2i, i2t)
    elif word_representation == 'b':
        return bm.SecondModel(w2i, t2i, i2t, c2i)
    elif word_representation == 'c':
        return bm.ThirdModel(w2i, t2i, i2t, p2i, s2i)
    else:
        return bm.FourthModel(w2i, t2i, i2t, c2i)


def load_dictionaries():
    """
    loading the model's dictionaries
    :return: model's dictionaries
    """
    with open('data.pkl', 'rb') as output:
        w2i = pickle.load(output)
        t2i = pickle.load(output)
        i2t = pickle.load(output)
        c2i = pickle.load(output)
        p2i = pickle.load(output)
        s2i = pickle.load(output)
    return [w2i, t2i, i2t, c2i, p2i, s2i]


def load_model(model):
    """
    populating the model
    :param model: model to populate
    :return:
    """
    model.model.populate('model')


def load(word_representation, model_file_path):
    """
    loading all the model data from the zip file including dictionaries and creating specific model
    :param word_representation: user's input for word representation
    :param model_file_path: model's file path
    :return: specific model
    """
    with ZipFile(model_file_path, 'r') as zip_file:
        zip_file.extractall()
    dictionaries = load_dictionaries()
    model = create_model(word_representation, dictionaries)
    load_model(model)
    os.remove("data.pkl")
    os.remove("model")
    return model


def main(argv):
    time.sleep(1)  # wait 1 sec for dy packages to be allocated and loaded to memory
    word_representation = argv[0]
    model_file_name = argv[1]
    input_filename = argv[2]
    type_of_data_set = argv[3]
    input_file_path = '{}/{}'.format(type_of_data_set, input_filename)
    model_file_path = '{}/{}_{}'.format(type_of_data_set, word_representation, model_file_name)
    output_file_path = '{}/{}'.format(type_of_data_set, word_representation)
    print('generating the data...')
    blind_data = ut.generate_validation_data(input_file_path)
    print('creating the model...')
    model = load(word_representation, model_file_path)
    print('predicting...')
    predicted_outputs = predicting(blind_data, model)
    print('writing the predictions...')
    ut.output_predicted_tags(predicted_outputs, blind_data, output_file_path)


if __name__ == '__main__':
    main(sys.argv[1:])
