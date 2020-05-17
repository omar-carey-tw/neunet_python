from svc.net import *
from svc.config import data_amount, l_nodes, training_iter

import matplotlib.pyplot as plt
from helpers.data import data


if __name__ == '__main__':

    save_epoch = True
    epoch = 10
    images = data[0]
    labels = data[1]

    if epoch is not None:
        print('<<< Starting Epoch Training >>>')
        learn_rate = 0.5

        neu_net = NeuNetBuilder(l_nodes).act("relu").cost("expquadratic").build()

        epoch_acc = []
        epoch_cost = []
        for i in range(epoch):

            cost = neu_net.train(images,
                                 labels,
                                 training_iter,
                                 learn_rate=0.5,
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

        if save_epoch:
            pickle.dump(neu_net, open('train_epoch_20', 'wb'))

    else:

        learn_rate = 0.5

        neu_net = NeuNetBuilder(l_nodes).act("relu").cost("expquadratic").build()
        cost = neu_net.train(images,
                             labels,
                             training_iter,
                             learn_rate=learn_rate,
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
