import random
import pytest

from svc.net import *

# python3 -m cProfile -s tottime svc/main.py
class TestNet:

    def test_neunet_instance(self):
        l_nodes = np.random.randint(low=1, high=10, size=random.randint(5, 10))

        neu_net = NeuNetBuilder(list(l_nodes)).act("sigmoid").cost("quadratic").build()

        assert neu_net.layers == len(l_nodes)
        assert len(neu_net.weights) == neu_net.layers - 1
        assert len(neu_net.bias) == neu_net.layers - 1

        for i in range(len(l_nodes)):
            assert neu_net.l_nodes[i] == l_nodes[i]

    def test_eval_sigmoid(self):
        l_nodes = np.random.randint(low=1, high=10, size=random.randint(5, 10))

        neu_net = NeuNetBuilder(list(l_nodes)).act("sigmoid").cost("quadratic").build()

        data = np.random.uniform(size=(l_nodes[0], 1))
        a_l = neu_net.evaluate(data)

        for i in range(len(a_l)):
            assert a_l[i].shape[0] == l_nodes[i]
            assert a_l[i].shape[1] == 1, f'l_nodes: {l_nodes} shape:{a_l[i].shape}'

    def test_eval_relu(self):
        l_nodes = np.random.randint(low=1, high=10, size=random.randint(5, 10))
        neu_net = NeuNetBuilder(list(l_nodes)).act("relu").cost("quadratic").build()

        data = np.random.uniform(size=(l_nodes[0], 1))

        a_l = neu_net.evaluate(data)

        for i in range(len(a_l)):
            assert a_l[i].shape[0] == l_nodes[i]
            assert a_l[i].shape[1] == 1, f'l_nodes: {l_nodes} shape:{a_l[i].shape}'

    def test_train_sigmoid(self):
        from tests.config_test import test_data_amount, test_training_iter, test_l_nodes
        neu_net = NeuNetBuilder(list(test_l_nodes)).act("sigmoid").cost("quadratic").build()

        train_data, train_labels = self.get_test_data(test_data_amount, test_l_nodes)
        cost_testing = neu_net.train(train_data, train_labels, test_training_iter, learn_rate=0.1, save=True)

        self.cleanup_files(test_data_amount, test_training_iter)
        assert cost_testing[1][-1] <= cost_testing[1][0]

    def test_train_relu(self):
        from tests.config_test import test_data_amount, test_training_iter, test_l_nodes
        neu_net = NeuNetBuilder(list(test_l_nodes)).act("relu").cost("quadratic").build()

        train_data, train_labels = self.get_test_data(test_data_amount, test_l_nodes)
        cost_testing = neu_net.train(train_data, train_labels, test_training_iter, learn_rate=0.01, save=True,
                                     reg_constant=0.5)

        self.cleanup_files(test_data_amount, test_training_iter)
        assert cost_testing[1][-1] <= cost_testing[1][0]

    def test_train_relu_cubic(self):
        from tests.config_test import test_data_amount, test_training_iter, test_l_nodes
        neu_net = NeuNetBuilder(list(test_l_nodes)).act("relu").cost("cubic").build()

        train_data, train_labels = self.get_test_data(test_data_amount, test_l_nodes)
        cost_testing = neu_net.train(train_data, train_labels, test_training_iter, learn_rate=0.01, save=True,
                                     reg_constant=0.5)

        self.cleanup_files(test_data_amount, test_training_iter)
        assert cost_testing[1][-1] <= cost_testing[1][0]

    @pytest.mark.flaky(max_runs=10)
    def test_train_relu_expquadratic(self):
        from tests.config_test import test_data_amount, test_training_iter, test_l_nodes
        neu_net = NeuNetBuilder(list(test_l_nodes)).act("relu").cost("expquadratic").build()

        train_data, train_labels = self.get_test_data(test_data_amount, test_l_nodes)
        cost_testing = neu_net.train(train_data, train_labels, test_training_iter, learn_rate=0.01, save=True,
                                     reg_constant=0.5)

        self.cleanup_files(test_data_amount, test_training_iter)
        assert cost_testing[1][-1] <= cost_testing[1][0]

    @pytest.mark.flaky(max_runs=10)
    def test_train_relu_expquadratic_dropout(self):
        from tests.config_test import test_data_amount, test_training_iter, test_l_nodes
        neu_net = NeuNetBuilder(list(test_l_nodes)).act("relu").cost("expquadratic").build()

        train_data, train_labels = self.get_test_data(test_data_amount, test_l_nodes)
        cost_testing = neu_net.train(train_data, train_labels, test_training_iter, learn_rate=0.01, save=True)

        self.cleanup_files(test_data_amount, test_training_iter)
        assert cost_testing[1][-1] <= cost_testing[1][0]


    def cleanup_files(self, test_data_amount, test_training_iter):

        path = (os.getcwd() + '/svc/train_objects/').replace('tests/', '')
        mnistobj = "mnistobj_iter_" + str(test_training_iter) + "_data_" + str(test_data_amount)
        mnistcost = "mnistcost_iter_" + str(test_training_iter) + "_data_" + str(test_data_amount)
        mnistacc = "mnistacc_iter_" + str(test_training_iter) + "_data_" + str(test_data_amount)

        os.remove(path + mnistacc)
        os.remove(path + mnistcost)
        os.remove(path + mnistobj)

    def get_test_data(self, test_data_amount, test_l_nodes):
        train_data = [0] * test_data_amount
        train_labels = [0] * test_data_amount

        for i in range(test_data_amount):
            train_data[i] = np.random.uniform(size=(test_l_nodes[0], 1))
            train_labels[i] = np.random.uniform(size=(test_l_nodes[-1], 1))

        return train_data, train_labels
