from svc.net import *
from svc.config import data_amount, l_nodes, probability, training_iter

import matplotlib.pyplot as plt
from helpers.data import data


if __name__ == '__main__':

    epoch = 30
    images = data[0][0:data_amount]
    labels = data[1][0:data_amount]

    if epoch is not None:

        neu_net = NeuNetBuilder(l_nodes).act("relu").cost("quadratic").build()

        epoch_acc = []
        epoch_cost = []
        for i in range(epoch):

            cost = neu_net.train(images,
                                 labels,
                                 training_iter,
                                 learn_rate=0.15,
                                 reg_constant=0.001,
                                 probability=probability,
                                 save=False
                                 )

            epoch_acc.append(cost[2][-1])
            epoch_cost.append(cost[1][-1])
            print(f'Epoch: {i}')

        plt.plot(list(range(epoch)), epoch_acc)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.show()

        plt.plot(list(range(epoch)), epoch_cost)
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()

    else:

        neu_net = NeuNetBuilder(l_nodes).act("relu").cost("cubic").build()

        cost = neu_net.train(images,
                             labels,
                             training_iter,
                             learn_rate=0.15,
                             reg_constant=0.001,
                             probability=probability,
                             save=False
                             )

        plt.plot(list(range(training_iter)), cost[1])
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.show()

        plt.plot(list(range(training_iter)), cost[2])
        plt.xlabel("Iterations")
        plt.ylabel("Acc")
        plt.show()
