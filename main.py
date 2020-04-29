from svc.net import *
from svc.config import data_amount

import matplotlib.pyplot as plt
from helpers.data import data


if __name__ == '__main__':

    l_nodes = [784, 20, 10]

    neu_net = NeuNetBuilder(l_nodes).act("relu").cost("quadratic").build()

    training_iter = 10
    probability = 0.5

    images = data[0][0:data_amount]
    labels = data[1][0:data_amount]

    cost = neu_net.train(images, labels, training_iter, learn_rate=0.1, save=False, probability=probability)

    plt.plot(list(range(training_iter)), cost[1])
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()

    plt.plot(list(range(training_iter)), cost[2])
    plt.xlabel("Iterations")
    plt.ylabel("Acc")
    plt.show()
