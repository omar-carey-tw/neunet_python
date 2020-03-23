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
        "low and up are upper and lower bounds for uniform distribution"

        self.l_nodes = l_nodes
        self.layers = len(l_nodes)

        if self.layers < 2:
            print("please create neural net with more than 2 layers!")
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
            Evaluates neural net output based off of existing weights
        """

        if len(inputs) != self.l_nodes[0]:
            print("input size not compatible with network")
        else:

            a_l = [0]*self.layers
            a_l[0] = self.act(inputs)

            for i in range(1,self.layers):
                a_l[i] = np.dot(self.weights[i-1], self.act(a_l[i-1])) + self.bias[i-1]

        return a_l


    def train(self,data,expect,learn_rate = 0.5):
        """ 
            Trains neural net by backpropagation using given data:
                is a list of lists where each list contains data
                expect is an array of the expected data
        """

        for ii in range(len(data)):
            self.eval(ii)

    



    
    def act(self, y):
        # sigmoid
        return 1/(1+np.exp(-y))

    def dact(self,y):
        # derivative of sigmoid
        return self.act(y)*(1-self.act(y))

    
        