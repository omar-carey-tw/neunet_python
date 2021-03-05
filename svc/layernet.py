from svc.layerfactory import BuildLayerTypes

import numpy as np


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

    def train(self, data_set, label_set):

        training_constant = 1

        for i, data in enumerate(data_set):

            label = label_set[i]

            a_l = self.eval(data)
            z_l = self.eval_weighted(data)

            delta_output = self.cost_layer.dcostdact(a_l[-1], label) * self.layer_types[-1].dactdz(z_l[-1])

            self.back_propragate(a_l, z_l, delta_output, training_constant)

    def eval(self, data):

        a_l = [0.0]*self.layers
        a_l[0] = self.layer_types[0].act(data)

        for i in range(1, self.layers):
            dot_product = np.dot(self.weights[i - 1], a_l[i - 1]) + self.bias[i - 1]
            a_l[i] = self.layer_types[i].act(dot_product)

        return a_l

    def eval_weighted(self, data):

        z_l = [0.0]*self.layers
        z_l[0] = data

        for i in range(1, self.layers):
            dot_product = np.dot(self.weights[i - 1], z_l[i - 1]) + self.bias[i - 1]
            z_l[i] = dot_product

        return z_l

    def back_propragate(self, a_l, z_l, delta_l, training_constant):

        for i in range(self.layers - 2, -1, -1):

            delta_weights = np.dot(delta_l, a_l[i].transpose())
            self.bias[i] -= training_constant * delta_l

            delta_l = np.dot(self.weights[i].transpose(), delta_l) * self.layer_types[i].dactdz(z_l[i])
            self.weights[i] -= training_constant * delta_weights


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
                or self.net.cost_layer is None:

            raise NameError("One of your constructors is empty")

        elif len(self.net.layer_types) != len(self.net.l_nodes):
            raise NameError("Lengths of layer types and number of layers do not match")

        elif self.net.layers < 2:
            raise NameError("Network too short, provide more than 2 layers!!")

        return self.net
