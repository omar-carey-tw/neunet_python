import numpy as np
import math
import random

from typing import *


class NeuNet:
    """
        Attributes:
        l_noes: list of int representing number of nodes per layer
        layers: number of layers including output and input
        weights: list of arrays of weight matrices
    """

    def __init__(self,l_nodes: List[int]) -> None:

        self.l_nodes = l_nodes
        self.layers = len(l_nodes)

        if self.layers < 2:
            print("please create neural net with more than 2 layers!")
            return
        else:
            _weights = []
            _bias = []

            for i in range(1,self.layers):
                _weights.append(np.random.uniform(size=(self.l_nodes[i],self.l_nodes[i-1])))
                _bias.append(np.random.uniform(size=(self.l_nodes[i],1)))

        self.weights = _weights
        self.bias = _bias


    def eval(self, inputs: List[float]) -> List[np.array]:
        """
            Evaluates activation layers
        """

        if len(inputs) != self.l_nodes[0]:
            print("input size not compatible with network")
            return 
        else:

            a_l = [0]*self.layers
            a_l[0] = self.act(inputs)

            for i in range(1,self.layers):
                a_l[i] = np.dot(self.weights[i-1], self.act(a_l[i-1])) + self.bias[i-1]

        return a_l

    def eval_weighted(self, inputs: List[float]) -> List[np.array]:
        """
            Evaluates weighted sum layers
        """

        if len(inputs) != self.l_nodes[0]:
            print("input size not compatible with network")
            return 
        else:

            z_l = [0]*self.layers
            z_l[0] = inputs

            for i in range(1,self.layers):
                z_l[i] = np.dot(self.weights[i-1], z_l[i-1]) + self.bias[i-1]

        return z_l


    def train(self, train_data: List[np.array] ,train_labels: List[np.array] ,training_iter: int, learn_rate = 0.5) -> List[float]:
        """ 
            Trains neural net by backpropagation using given data:
                is a list of lists where each list contains data
                expect is an array of the expected data
        """
        Cost = [0.0] * training_iter

        for iter in range(training_iter):
            cost_iter = 0

            for i, data in enumerate(train_data):

                a_l = self.eval(data)
                z_l = self.eval_weighted(data)


                self.__back_prop(a_l, z_l, train_labels[i], learn_rate, train_batch_size=len(train_data))
        

        return Cost


    def __back_prop(self, act_layers: List[np.array], weight_layers: List[np.array], train_label,
     learning_rate: float, train_batch_size: int):

        layer_error = self.__output_error(act_layers[-1], weight_layers[-1], train_label)
        weight_error = np.dot(layer_error, act_layers[-2].transpose())

        self.weights[-1] = self.weights[-1] - weight_error * (learning_rate / train_batch_size)
        self.bias[-1] = self.bias[-1] - layer_error * (learning_rate / train_batch_size)


         



    def __output_error(self,output_act: np.array, output_weighted: np.array, train_label: np.array) -> np.array:

        return self.dcost(output_act, train_label)*self.dact(output_weighted)


    def act(self, y):
        # sigmoid
        return 1/(1+np.exp(-y))

    def dact(self,y):
        # derivative of sigmoid
        return self.act(y)*(1-self.act(y))

    def cost(self, output_act: np.array, training_label) -> np.array:
        # cost for parabolic 
        return 0.5*(output_act - training_label)**2

    def dcost(self, output_act: np.array, training_label) -> np.array:
        # derivative of parabolic cost
        return output_act - training_label


    
        