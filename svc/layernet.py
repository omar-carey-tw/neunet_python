from svc.layerfactory import BuildLayerTypes
from helpers.helpers import pickle_object, check_file, pickle_load, generate_mask

import numpy as np
import os

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__)).replace("/svc", "")

# todo: Add a new more efficient cost function and tests
# todo: delete old neunet and testneunet files


class LayerNeuNet:
    """
        Attributes:
        l_nodes: list of int representing number of nodes per layer
        layers: number of layers including output and input
        layer_types: type of activation function associated with each layer
        weights: weigths in neural network
        bias: bias in neural network
    """

    def __init__(self):

        self.l_nodes = None
        self.layer_types = None
        self.layers = None
        self.weights = None
        self.bias = None
        self.cost_layer = None
        self.accuracy = []
        self.cost = []

    def evaluate(self, inputs):
        """"
            Evaluates activation layers, single pass with no mask
        """

        if len(inputs) != self.l_nodes[0]:
            raise NameError("Your data input is not same length as first layer of neural network")

        a_l = [0.0]*self.layers
        a_l[0] = self.layer_types[0].act(inputs)

        for i in range(1, self.layers):
            dot_product = np.dot(self.weights[i - 1], a_l[i - 1]) + self.bias[i - 1]
            a_l[i] = self.layer_types[i].act(dot_product)

        return a_l

    def train(self, data_set, label_set, training_iterations, learning_rate=0.5, save=False, reg_const=0, probability=None):

        file_name_list = ["mnist_obj_iter", f"{training_iterations}", f"data_{len(data_set)}",
                          f"learning_rate_{learning_rate}"]
        directory = os.path.join("svc", "trained_objects")

        if check_file(directory, file_name_list):
            trained_network = pickle_load(directory, file_name_list)
            return trained_network

        else:

            mask = generate_mask(self.l_nodes, len(data_set), training_iterations, probability)

            self.cost = np.zeros(shape=(training_iterations, 1))
            self.accuracy = np.zeros(shape=(training_iterations, 1))

            training_constant = learning_rate / len(data_set)

            for index in range(training_iterations):

                accuracy_iter = 0
                cost_iter = 0

                for i, data in enumerate(data_set):

                    label = label_set[i]

                    a_l = self.eval(data, mask[index][i])
                    z_l = self.eval_weighted(data, mask[index][i])

                    accuracy_iter += self.check_accuracy(label, a_l[-1])
                    cost_iter += self.cost_layer.cost(a_l[-1], label, weight=self.weights[-1], reg_const=0)

                    delta_output = self.cost_layer.dcostdact(a_l[-1], label) * self.layer_types[-1].dactdz(z_l[-1])

                    self.back_propragate(a_l, z_l, delta_output, training_constant, reg_const=reg_const)

                self.cost[index] = cost_iter / len(data_set)
                self.accuracy[index] = accuracy_iter / len(data_set)

            if save:
                pickle_object(directory, file_name_list, self)

            return self

    def eval(self, data, mask):

        a_l = [0.0]*self.layers
        a_l[0] = self.layer_types[0].act(data)

        for i in range(1, self.layers):
            dot_product = np.dot(self.weights[i - 1], a_l[i - 1]) + self.bias[i - 1]
            a_l[i] = self.layer_types[i].act(dot_product) * mask[i]

        return a_l

    def eval_weighted(self, data, mask):

        z_l = [0.0]*self.layers
        z_l[0] = data

        for i in range(1, self.layers):
            dot_product = np.dot(self.weights[i - 1], z_l[i - 1]) + self.bias[i - 1]
            z_l[i] = dot_product * mask[i]

        return z_l

    def back_propragate(self, a_l, z_l, delta_l, training_constant, reg_const):

        for i in range(self.layers - 2, -1, -1):

            delta_weights = np.dot(delta_l, a_l[i].transpose()) + reg_const * self.weights[i]
            self.bias[i] -= training_constant * delta_l

            delta_l = np.dot(self.weights[i].transpose(), delta_l) * self.layer_types[i].dactdz(z_l[i])
            self.weights[i] -= training_constant * delta_weights

    def check_accuracy(self, train_label, output: np.array, tol=0.01):
        index = np.argmax(train_label)
        delta = np.abs(output[index] - train_label[index])

        return int(delta < tol)


class LayerNeuNetBuilder:

    def __init__(self):

        self.buildlayers = BuildLayerTypes()
        self.net = LayerNeuNet()

    def set_layers(self, layer_types):

        self.net.layer_types = self.buildlayers.build_activation_layers(layer_types)
        return self

    def set_weights_and_bias(self, l_nodes):

        weights_and_bias = self.buildlayers.build_weights_and_bias(l_nodes)

        self.net.weights = weights_and_bias.get("weights")
        self.net.bias = weights_and_bias.get("bias")
        self.net.l_nodes = l_nodes
        self.net.layers = len(l_nodes)

        return self

    def set_cost(self, cost_type):

        cost_layer = self.buildlayers.build_cost_layer(cost_type)

        self.net.cost_layer = cost_layer

        return self

    def build(self):

        if self.net.l_nodes is None \
                or self.net.layer_types is None \
                or self.net.layers is None \
                or self.net.weights is None \
                or self.net.bias is None \
                or self.net.cost_layer is None \
                or self.net.accuracy != [] \
                or self.net.cost != []:

            raise NameError("One of your constructors is empty")

        elif len(self.net.layer_types) != len(self.net.l_nodes):
            raise NameError("Lengths of layer types and number of layers do not match")

        elif self.net.layers < 2:
            raise NameError("Network too short, provide more than 2 layers!!")

        return self.net
