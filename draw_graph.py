import pickle
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import collections

STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}


def load_graph(file_path):
    """
    loading the graph from the file
    :param file_path: file relative path
    :return: graph as ordered sorted dict
    """
    with open(file_path, 'rb') as dicts_file:
        graph = pickle.load(dicts_file)
    return collections.OrderedDict(sorted(graph.items()))


# generating pos graphs and saving as 1 plot
a_pos = load_graph('pos/a_graph.pkl')
b_pos = load_graph('pos/b_graph.pkl')
c_pos = load_graph('pos/c_graph.pkl')
d_pos = load_graph('pos/d_graph.pkl')
label1, = plt.plot(a_pos.keys(), a_pos.values(), "r-", label='a - pos')
label2, = plt.plot(b_pos.keys(), b_pos.values(), "g-", label='b - pos')
label3, = plt.plot(c_pos.keys(), c_pos.values(), "b-", label='c - pos')
label4, = plt.plot(d_pos.keys(), d_pos.values(), "y-", label='d - pos')
plt.legend(handler_map={label1: HandlerLine2D(numpoints=4)})
plt.ylabel("accuracy")
plt.xlabel("iter number / 100")
plt.show()

# generating ner graphs and saving as 1 plot
a_ner = load_graph('ner/a_graph.pkl')
b_ner = load_graph('ner/b_graph.pkl')
c_ner = load_graph('ner/c_graph.pkl')
d_ner = load_graph('ner/d_graph.pkl')
label1, = plt.plot(a_ner.keys(), a_ner.values(), "r-", label='a - ner')
label2, = plt.plot(b_ner.keys(), b_ner.values(), "g-", label='b - ner')
label3, = plt.plot(c_ner.keys(), c_ner.values(), "b-", label='c - ner')
label4, = plt.plot(d_ner.keys(), d_ner.values(), "y-", label='d - ner')
plt.legend(handler_map={label1: HandlerLine2D(numpoints=4)})
plt.ylabel("accuracy")
plt.xlabel("iter number / 100")
plt.show()
