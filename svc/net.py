import numpy as np
import numpy.linalg as la
import dill as pickle

from typing import *
from helpers.helpers import generate_mask, check_previous_train, pickle_meta_data

# todo: look into making each layer independent activation (class and interfaces for layers)
# todo: write test script to find best constant values for train
# todo: Stochastic gradient descent
# todo: convolution neural network
# todo: implement adaptive learning rate

# I guess a rule of thumb is to use dropout over matrix reg for large networks (allegedly)


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
            raise NameError('Please create network of atleast 2 layers')
        else:
            _weights = []
            _bias = []

            for i in range(1, self.layers):
                he_regularization = np.sqrt(1 / self.l_nodes[i-1])

                _weights.append(np.random.randn(self.l_nodes[i], self.l_nodes[i-1])
                                * he_regularization)
                _bias.append(np.zeros(shape=(self.l_nodes[i], 1)))

        self.weights = _weights
        self.bias = _bias

    def train(self, train_data: List[np.array], train_labels: List[np.array], training_iter: int, probability=None,
              learn_rate=0.5, reg_constant=0, save=False) -> np.array:
        """
            Trains neural net by backpropagation using given helpers:
                is a list of lists where each list contains helpers
                expect is an array of the expected helpers

        """
        check_previous_run = check_previous_train(training_iter, len(train_data), probability, learn_rate)
        if check_previous_run:
            return check_previous_run
        else:
            mask = generate_mask(self.l_nodes, len(train_data), training_iter, probability)

            cost = np.zeros(shape=(training_iter, 1))
            acc = np.zeros(shape=(training_iter, 1))
            train_batch_size = len(train_data)

            for index in range(training_iter):
                cost_iter = 0
                acc_iter = 0

                for i, data in enumerate(train_data):
                    a_l = self.__eval(data, mask[index][i])
                    z_l = self.__eval_weighted(data, mask[index][i])

                    acc_iter += self.accuracy(train_labels[i], a_l[-1])
                    cost_iter += self.cost(a_l[-1], train_labels[i], self.weights[-1], reg_constant)

                    self.back_prop(a_l, z_l, train_labels[i], learn_rate, reg_constant, train_batch_size)

                cost[index] = cost_iter / train_batch_size
                acc[index] = acc_iter / train_batch_size


            #todo: consider dumping all three things into one folder (/train_objects/<file>)
            if save:
                pickle_obj, pickle_cost, pickle_acc, path_to_obj = pickle_meta_data(training_iter, len(train_data),
                                                                                    probability, learn_rate)
                pickle.dump(self, open(path_to_obj + pickle_obj, 'wb'))
                pickle.dump(cost, open(path_to_obj + pickle_cost, 'wb'))
                pickle.dump(acc, open(path_to_obj + pickle_acc, 'wb'))

            result = {
                "net": self,
                "cost": cost,
                "accuracy": acc
            }

            return result

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

    def evaluate(self, inputs: np.array) -> np.array:
        """
            Evaluates activation layers, single pass with no mask
        """

        a_l = [0.0]*self.layers
        a_l[0] = self.act(inputs)

        for i in range(1, self.layers):
            a_l[i] = self.act(np.dot(self.weights[i - 1], a_l[i - 1]) + self.bias[i - 1])

        return a_l

    def __eval(self, inputs: np.array, mask) -> np.array:
        """
            Evaluates activation layers
        """

        a_l = [0.0]*self.layers
        a_l[0] = self.act(inputs)

        for i in range(1, self.layers):
            a_l[i] = (self.act(np.dot(self.weights[i - 1], a_l[i - 1]) + self.bias[i - 1])) * mask[i]

        return a_l

    def __eval_weighted(self, inputs: List[float], mask) -> np.array:
        """
            Evaluates weighted sum layers
        """

        z_l = [0]*self.layers
        z_l[0] = inputs

        for i in range(1, self.layers):
            z_l[i] = (np.dot(self.weights[i - 1], z_l[i - 1]) + self.bias[i - 1]) * mask[i]

        return z_l

    def output_error(self, output_act: np.array, output_weighted: np.array, train_label: np.array) -> np.array:
        return self.dcost(output_act, train_label) * self.dact(output_weighted)

    def dcostw(self, layer_error, act_layer, weight, reg_const):
        result = np.dot(layer_error, act_layer.transpose()) + reg_const * weight
        return result

    def accuracy(self, train_label, output: np.array):
        index = np.argmax(train_label)
        delta = np.abs(output[index] - train_label[index])[0]
        tol = 0.01

        return int(delta < tol)

    # Setters

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

    #todo: <<<<< These derivatives are with respect to activation (no action) >>>>
    def act(self, act_function):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def dsigmoid(x):
            return np.exp(-x) / (1 + np.exp(-x)) ** 2

        def relu(x):
            return np.maximum(np.zeros(shape=(len(x), 1)), x)

        def drelu(x):
            return np.greater_equal(x, np.zeros(shape=x.shape))

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

        def cubic(output_act, training_label, weight, reg_const):
            return sum(1/3*output_act**3 - 0.5*training_label*output_act**2) + \
                   reg_const / 2 * la.norm(weight, 2) ** 2

        def dcubic(output_act, training_label):
            return output_act**2 - training_label*output_act

        def expquadratic(output_act, training_label, weight, reg_const):
            return sum(1/2 * np.exp((output_act - training_label)**2) + \
                   reg_const/2 * la.norm(weight, 2) ** 2)

        def dexpquadratic(output_act, training_label):
            return (output_act - training_label) * np.exp((output_act - training_label)**2)

        if cost_function == "quadratic":
            self.net.set_cost(quadratic)
            self.net.set_dcost(dquadratic)
        elif cost_function == "cubic":
            self.net.set_cost(cubic)
            self.net.set_dcost(dcubic)
        elif cost_function == "expquadratic":
            self.net.set_cost(expquadratic)
            self.net.set_dcost(dexpquadratic)
        else:
            raise NameError('Please enter valid cost name')

        return self

    def build(self) -> NeuNet:

        return self.net





