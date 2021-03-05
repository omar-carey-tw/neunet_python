from typing import *
import numpy as np
import numpy.linalg as la

# todo: add checks to see if components actually got set (check to see if initated lists are still empty after  "setting"


class BuildLayerTypes:

    def build_activation_layers(self, layer_types: List[str]) -> List:
        layers = []

        for layer in layer_types:
            if layer == "relu":
                layers.append(ReLu())
            elif layer == "sigmoid":
                layers.append(Sigmoid())
        return layers

    def build_weights_and_bias(self, l_nodes: List[int]):

        _weights = []
        _bias = []

        for i in range(1, len(l_nodes)):
            he_regularization = np.sqrt(1 / l_nodes[i-1])

            _weights.append(np.random.randn(l_nodes[i], l_nodes[i-1])
                            * he_regularization)
            _bias.append(np.zeros(shape=(l_nodes[i], 1)))

        result = {
            "weights": _weights,
            "bias": _bias
        }
        return result

    def build_cost_layer(self, cost_type: str):

        cost_layer = None

        if cost_type == "quadratic":
            cost_layer = Cuadratic()

        return cost_layer


class ReLu:

    def act(self, x):
        result = np.maximum(np.zeros(shape=(len(x), 1)), x)
        return result

    def dactdz(self, x):
        result = np.greater_equal(x, np.zeros(shape=x.shape))
        return result


class Sigmoid:

    def act(self, x):
        result = 1 / (1 + np.exp(-x))
        return result

    def dactdz(self, x):
        result = np.exp(-x) / (1 + np.exp(-x)) ** 2
        return result


class Cuadratic:

    def cost(self, output_act, training_label, weight, reg_const):
        result = sum(0.5 * (output_act - training_label) ** 2) + \
                   reg_const/2 * la.norm(weight, 2) ** 2
        return result

    def dcostdact(self, output_act, training_label):
        result = output_act - training_label

        return result

