from svc.net import *

import matplotlib.pyplot as plt
from svc.data import data


if __name__ == '__main__':

    l_nodes = [784, 15, 10]
    neu_net = NeuNet(l_nodes)
    old_weights = neu_net.weights
    old_bias = neu_net.bias

    training_iter = 1
    data_amount = 1000

    images = data[0][0:data_amount]
    labels = data[1][0:data_amount]

    cost = neu_net.train(images, labels, training_iter)

    new_weights = neu_net.weights
    new_bias = neu_net.bias

    plt.plot(list(range(training_iter)), cost)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()

    print(new_weights[-1], '\n')
    print(old_weights[-1], '\n')


    # print(cost[0]-cost[-1])
    # must be issue in back_prop (maybe weights and biases arent updating correctly)
