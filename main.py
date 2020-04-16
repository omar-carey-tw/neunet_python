from svc.net import *
from svc.config import data_amount

import matplotlib.pyplot as plt
from svc.data import data


if __name__ == '__main__':

    l_nodes = [784, 5, 3, 10]

    neu_net = NeuNetBuilder(l_nodes).act("sigmoid").cost("quadratic").build()

    training_iter = 10

    images = data[0][0:data_amount]
    labels = data[1][0:data_amount]

    cost = neu_net.train(images, labels, training_iter)
    print(cost[1])

    plt.plot(list(range(training_iter)), cost[1])
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()


