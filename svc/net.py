import numpy as np
import numpy.linalg as la
from typing import *
import dill as pickle
import os

# todo: fix it being so slow now that dropout is a thing (or implement it into builder pattern)
    # todo: np.choice is hella slow
# todo: look into new cost functions
# todo: write test script to find best constant values for train

# I guess a rule of thumb is to use dropout over matrix reg for large networks (i guess)


class NeuNet:
    """
        Attributes:
        l_noes: list of int representing number of nodes per layer
        layers: number of layers including output and input
    """

    def __init__(self, l_nodes: List[int]) -> None:

        self.l_nodes = l_nodes
        self.layers = len(l_nodes)

        if self.layers < 2:
            print("please create neural net with more than 2 layers!")
            return
        else:
            _weights = []
            _bias = []

            for i in range(1, self.layers):
                he_regularization = np.sqrt(2 / (self.l_nodes[i] + self.l_nodes[i-1]))

                _weights.append(np.random.randn(self.l_nodes[i], self.l_nodes[i-1])
                                * he_regularization)
                _bias.append(np.random.randn(self.l_nodes[i], 1)
                             * he_regularization)

        self.weights = _weights
        self.bias = _bias

    def train(self, train_data: List[np.array], train_labels: List[np.array], training_iter: int, learn_rate=0.5,
              reg_constant=0, save=False, probability=0.5) -> np.array:
        """
            Trains neural net by backpropagation using given data:
                is a list of lists where each list contains data
                expect is an array of the expected data
        """

        pickle_obj = "mnistobj_" + "iter_" + str(training_iter) + "_data_" + str(len(train_data))
        pickle_cost = "mnistcost_" + "iter_" + str(training_iter) + "_data_" + str(len(train_data))
        pickle_acc = "mnistacc_" + "iter_" + str(training_iter) + "_data_" + str(len(train_data))

        if pickle_obj in os.listdir() and pickle_cost in os.listdir():
            print("Loading previous training run ...")
            neunet = pickle.load(open(pickle_obj, 'rb'))
            cost = pickle.load(open(pickle_cost, 'rb'))
            acc = pickle.load(open(pickle_acc, 'rb'))

            return neunet, cost, acc

        else:
            cost = np.zeros(shape=(training_iter, 1))
            acc = np.zeros(shape=(training_iter, 1))
            train_batch_size = len(train_data)

            for index in range(training_iter):
                cost_iter = 0
                acc_iter = 0

                for i, data in enumerate(train_data):
                    mask = []
                    for j in self.l_nodes:
                        mask.append(np.random.choice([1, 0], size=(j, 1), p=[probability, 1 - probability]).astype(np.bool))

                    a_l = self.eval_dropout(data, mask)
                    z_l = self.eval_weighted_dropout(data, mask)

                    acc_iter += self.accuracy(train_labels[i], a_l[-1])
                    cost_iter += self.cost(a_l[-1], train_labels[i], self.weights[-1], reg_constant)

                    self.back_prop(a_l, z_l, train_labels[i], learn_rate, reg_constant, train_batch_size)

                cost[index] = cost_iter / train_batch_size
                acc[index] = acc_iter / train_batch_size

            if save:
                pickle.dump(self, open(pickle_obj, 'wb'))
                pickle.dump(cost, open(pickle_cost, 'wb'))
                pickle.dump(acc, open(pickle_acc, 'wb'))

            return self, cost, acc

    def back_prop(self, act_layers: List[np.array], weight_layers: List[np.array], train_label,
                  learning_rate: float, reg_const: float, train_batch_size):

        const = learning_rate / train_batch_size
        layer_error = self.output_error(act_layers[-1], weight_layers[-1], train_label)
        weight_error = self.dcostw(layer_error, act_layers[-2], self.weights[-1], reg_const)

        self.weights[-1] -= weight_error * const
        self.bias[-1] -= layer_error * const

        for i in range(self.layers - 1, 1, -1):
            layer_error = self.dact(weight_layers[i - 1]) * np.dot(self.weights[i-1].transpose(), layer_error)
            weight_error = self.dcostw(layer_error, act_layers[i-2], self.weights[i-2], reg_const)

            self.weights[i - 2] -= weight_error * const
            self.bias[i - 2] -= layer_error * const

    def eval_dropout(self, inputs: np.array, mask) -> np.array:
        """
            Evaluates activation layers
        """

        a_l = [0.0]*self.layers
        temp = self.act(inputs)
        a_l[0] = np.zeros(shape=(self.l_nodes[0], 1))
        a_l[0][mask[0]] = temp[mask[0]]

        for i in range(1, self.layers):
            a_l[i] = np.zeros(shape=(self.l_nodes[i], 1))
            temp = self.act(np.dot(self.weights[i - 1], a_l[i - 1]) + self.bias[i - 1])
            a_l[i][mask[i]] = temp[mask[i]]

        return a_l

    def eval_weighted_dropout(self, inputs: List[float], mask) -> np.array:
        """
            Evaluates weighted sum layers
        """

        z_l = [0]*self.layers
        temp = inputs
        z_l[0] = np.zeros(shape=(self.l_nodes[0], 1))
        z_l[0][mask[0]] = temp[mask[0]]

        for i in range(1, self.layers):
            z_l[i] = np.zeros(shape=(self.l_nodes[i], 1))
            temp = np.dot(self.weights[i - 1], z_l[i - 1]) + self.bias[i - 1]
            z_l[i][mask[i]] = temp[mask[i]]

        return z_l

    def eval(self, inputs: np.array) -> np.array:
        """
            Evaluates activation layers
        """

        a_l = [0.0]*self.layers
        a_l[0] = self.act(inputs)

        for i in range(1, self.layers):
            a_l[i] = self.act(np.dot(self.weights[i - 1], a_l[i - 1]) + self.bias[i - 1])

        return a_l

    def eval_weighted(self, inputs: List[float]) -> np.array:
        """
            Evaluates weighted sum layers
        """

        z_l = [0]*self.layers
        z_l[0] = inputs

        for i in range(1, self.layers):
            z_l[i] = np.dot(self.weights[i - 1], z_l[i - 1]) + self.bias[i - 1]

        return z_l

    def output_error(self, output_act: np.array, output_weighted: np.array, train_label: np.array) -> np.array:
        return self.dcost(output_act, train_label) * self.dact(output_weighted)

    def dcostw(self, layer_error, act_layer, weight, reg_const):
        result = np.dot(layer_error, act_layer.transpose()) + reg_const * weight
        return result

    def accuracy(self, train_label, output: np.array):
        index = np.argmax(train_label)
        delta = np.abs(output[index] - train_label[index])

        if delta < .1:
            return 1
        else:
            return 0


    def set_act(self, func):
        self.act = func

    def set_dact(self, func):
        self.dact = func

    def set_cost(self, func):
        self.cost = func

    def set_dcost(self, func):
        self.dcost = func


class NeuNetBuilder:

    def __init__(self, l_nodes: List[int]):

        self.net = NeuNet(l_nodes)

    def act(self, act_function):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def dsigmoid(x):
            return np.exp(-x) / (1 + np.exp(-x)) ** 2

        def relu(x):
            return np.maximum(np.zeros(shape=(len(x), 1)), x)

        def drelu(x):
            return np.ones(shape=(len(x), 1))

        if act_function == "sigmoid":
            self.net.set_act(sigmoid)
            self.net.set_dact(dsigmoid)
        elif act_function == "relu":
            self.net.set_act(relu)
            self.net.set_dact(drelu)
        else:
            raise NameError('Please enter valid activation name')

        return self

    def cost(self, cost_function):

        def quadratic(output_act, training_label, weight, reg_const):
            return sum(0.5 * (output_act - training_label) ** 2) + \
                   reg_const/2 * la.norm(weight, 2) ** 2

        def dquadratic(output_act, training_label):
            return output_act - training_label

        if cost_function == "quadratic":
            self.net.set_cost(quadratic)
            self.net.set_dcost(dquadratic)
        else:
            raise NameError('Please enter valid cost name')

        return self

    def build(self) -> NeuNet:

        return self.net

