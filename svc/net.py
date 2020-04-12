import numpy as np
from typing import *
import dill as pickle
import os

# python3 -m cProfile -s tottime main.py
# todo: look into new cost functions


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
                _weights.append(np.random.uniform(size=(self.l_nodes[i], self.l_nodes[i - 1]), low=-1.0, high=1.0))
                _bias.append(np.random.uniform(size=(self.l_nodes[i], 1), low=-1.0, high=1.0))

        self.weights = _weights
        self.bias = _bias

    def eval(self, inputs: np.array) -> np.array:
        """
            Evaluates activation layers
        """

        a_l = [0.0]*self.layers
        a_l[0] = self.act(inputs)

        for i in range(1, self.layers):
            a_l[i] = self.act(np.matmul(self.weights[i - 1], a_l[i - 1]) + self.bias[i - 1])

        return a_l

    def eval_weighted(self, inputs: List[float]) -> np.array:
        """
            Evaluates weighted sum layers
        """

        z_l = [0]*self.layers
        z_l[0] = inputs

        for i in range(1, self.layers):
            z_l[i] = np.matmul(self.weights[i - 1], z_l[i - 1]) + self.bias[i - 1]

        return z_l

    def train(self, train_data: List[np.array], train_labels: List[np.array], training_iter: int, learn_rate=0.5) -> \
            np.array:
        """ 
            Trains neural net by backpropagation using given data:
                is a list of lists where each list contains data
                expect is an array of the expected data
        """

        pickle_obj = "mnistobj_" + "iter_" + str(training_iter) + "_data_" + str(len(train_data))
        pickle_cost = "mnistcost_" + "iter_" + str(training_iter) + "_data_" + str(len(train_data))

        if pickle_obj in os.listdir() and pickle_cost in os.listdir():
            print("Loading previous training run...")
            neunet = pickle.load(open(pickle_obj, 'rb'))
            cost = pickle.load(open(pickle_cost, 'rb'))

            return neunet, cost

        else:
            cost = np.zeros(shape=(training_iter, 1))
            train_batch_size = len(train_data)

            for index in range(training_iter):
                cost_iter = 0

                for i, data in enumerate(train_data):
                    a_l = self.eval(data)
                    z_l = self.eval_weighted(data)

                    cost_iter += np.sum(self.cost(a_l[-1], train_labels[i]))

                    self.back_prop(a_l, z_l, train_labels[i], learn_rate, train_batch_size)

                cost[index] = cost_iter / train_batch_size

            pickle.dump(self, open(pickle_obj, 'wb'))
            pickle.dump(cost, open(pickle_cost, 'wb'))

            return self, cost

    def back_prop(self, act_layers: List[np.array], weight_layers: List[np.array], train_label,
                  learning_rate: float, train_batch_size: int):

        layer_error = self.output_error(act_layers[-1], weight_layers[-1], train_label)
        weight_error = np.dot(layer_error, act_layers[-2].transpose())

        self.weights[-1] -= weight_error * (learning_rate / train_batch_size)
        self.bias[-1] -= layer_error * (learning_rate / train_batch_size)

        for i in range(self.layers - 1, 1, -1):
            layer_error = self.dact(weight_layers[i - 1]) * np.dot(self.weights[i - 1].transpose(), layer_error)
            weight_error = np.dot(layer_error, act_layers[i - 2].transpose())

            self.weights[i - 2] -= weight_error * (learning_rate / train_batch_size)
            self.bias[i - 2] -= layer_error * (learning_rate / train_batch_size)

    def output_error(self, output_act: np.array, output_weighted: np.array, train_label: np.array) -> np.array:

        return self.dcost(output_act, train_label) * self.dact(output_weighted)

    def set_act(self, func):
        self.act = func

    def set_dact(self, func):
        self.dact = func

    def set_cost(self, func):
        self.cost = func

    def set_dcost(self, func):
        self.dcost = func


class NeuNetBuilder:

    def __init__(self, l_nodes: List[int], cost_function: str, act_function: str) -> None:

        self.l_nodes = l_nodes
        self.cost_function = cost_function
        self.act_function = act_function

    def build(self) -> NeuNet:

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def dsigmoid(x):
            return np.exp(-x) / (1 + np.exp(-x)) ** 2


        def quadratic(output_act, training_label):
            return 0.5 * (output_act - training_label) ** 2

        def dquadratic(output_act, training_label):
            return output_act - training_label

        neunet = NeuNet(self.l_nodes)

        if self.act_function == "sigmoid":
            neunet.set_act(sigmoid)
            neunet.set_dact(dsigmoid)

        if self.cost_function == "quadratic":
            neunet.set_cost(quadratic)
            neunet.set_dcost(dquadratic)

        return neunet

