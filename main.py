from svc.net import *

import matplotlib.pyplot as plt
from svc.data import data


if __name__ == '__main__':

    l_nodes = [784, 5, 3, 10]
    neu_net = NeuNetBuilder(l_nodes, 'quadratic', "sigmoid").build()

    training_iter = 100
    data_amount = 100

    images = data[0][0:data_amount]
    labels = data[1][0:data_amount]

    cost = neu_net.train(images, labels, training_iter)

    plt.plot(list(range(training_iter)), cost[1])
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()


