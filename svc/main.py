from svc.net import *
from svc.config import data_amount, l_nodes, probability, training_iter

import matplotlib.pyplot as plt
from helpers.data import data


if __name__ == '__main__':

    neu_net = NeuNetBuilder(l_nodes).act("relu").cost("quadratic").build()

    images = data[0][0:data_amount]
    labels = data[1][0:data_amount]

    cost = neu_net.train(images, labels, training_iter, learn_rate=0.1, save=False, reg_constant=0.1, probability=probability)

    plt.plot(list(range(training_iter)), cost[1])
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()

    plt.plot(list(range(training_iter)), cost[2])
    plt.xlabel("Iterations")
    plt.ylabel("Acc")
    plt.show()
