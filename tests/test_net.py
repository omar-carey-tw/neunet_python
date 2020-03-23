import pytest
import numpy as np
import random

from svc.net import *


class TestNet:

    def test_it_works(self):
        assert(True)

    def test_neunet_instance(self):
        l_nodes = np.random.randint(low=1, high=10, size=random.randint(5,10))

        neu_net = NeuNet(l_nodes)

        assert neu_net.layers == len(l_nodes)
        assert len(neu_net.weights) == neu_net.layers - 1
        assert len(neu_net.bias) == neu_net.layers - 1


        for i in range(len(l_nodes)):
            assert neu_net.l_nodes[i] == l_nodes[i]

    def test_eval(self):
        l_nodes = np.random.randint(low=1, high=10, size=random.randint(5,10))
        neu_net = NeuNet(l_nodes)
        
        data = np.random.uniform(size=l_nodes[0])
        a_l = neu_net.eval(data)

        for i in range(len(a_l)):
            assert len(a_l[i]) == l_nodes[i]