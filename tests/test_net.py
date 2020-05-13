import random

from svc.net import *


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
        l_nodes = np.random.randint(low=1, high=10, size=random.randint(5, 10))
        neu_net = NeuNetBuilder(list(l_nodes)).act("sigmoid").cost("quadratic").build()

        amount_data = 50
        training_iter = 100

        path = os.getcwd() + '/svc/train_objects/'
        path.replace('tests/', '')
        mnistobj = "mnistobj_iter_" + str(training_iter) + "_data_" + str(amount_data)
        mnistcost = "mnistcost_iter_" + str(training_iter) + "_data_" + str(amount_data)
        mnistacc = "mnistacc_iter_" + str(training_iter) + "_data_" + str(amount_data)

        if mnistobj and mnistcost and mnistacc in os.listdir(path):

            os.remove(path + mnistacc)
            os.remove(path + mnistcost)
            os.remove(path + mnistobj)

        train_data = [0] * amount_data
        train_labels = [0] * amount_data

        for i in range(amount_data):
            train_data[i] = np.random.uniform(size=(l_nodes[0], 1))
            train_labels[i] = np.random.uniform(size=(l_nodes[-1], 1))

        cost_testing = neu_net.train(train_data, train_labels, training_iter, learn_rate=0.1, save=True)

        os.remove(path + mnistacc)
        os.remove(path + mnistcost)
        os.remove(path + mnistobj)

        assert cost_testing[1][-1] <= cost_testing[1][0]

    def test_train_relu(self):
        l_nodes = np.random.randint(low=1, high=10, size=random.randint(5, 10))
        neu_net = NeuNetBuilder(list(l_nodes)).act("relu").cost("quadratic").build()

        amount_data = 50
        training_iter = 100

        path = os.getcwd() + '/svc/train_objects/'
        path.replace('tests/', '')
        mnistobj = "mnistobj_iter_" + str(training_iter) + "_data_" + str(amount_data)
        mnistcost = "mnistcost_iter_" + str(training_iter) + "_data_" + str(amount_data)
        mnistacc = "mnistacc_iter_" + str(training_iter) + "_data_" + str(amount_data)

        if mnistobj and mnistcost and mnistacc in os.listdir(path):

            os.remove(path + mnistacc)
            os.remove(path + mnistcost)
            os.remove(path + mnistobj)

        train_data = [0] * amount_data
        train_labels = [0] * amount_data

        for i in range(amount_data):
            train_data[i] = np.random.uniform(size=(l_nodes[0], 1))
            train_labels[i] = np.random.uniform(size=(l_nodes[-1], 1))

        cost_testing = neu_net.train(train_data, train_labels, training_iter, learn_rate=0.01, save=True,
                                     reg_constant=0.5)

        os.remove(path + mnistacc)
        os.remove(path + mnistcost)
        os.remove(path + mnistobj)

        assert cost_testing[1][-1] <= cost_testing[1][0]

    def test_train_relu_cubic(self):
        l_nodes = np.random.randint(low=1, high=10, size=random.randint(5, 10))
        neu_net = NeuNetBuilder(list(l_nodes)).act("relu").cost("cubic").build()

        amount_data = 50
        training_iter = 100

        path = os.getcwd() + '/svc/train_objects/'
        path.replace('tests/', '')
        mnistobj = "mnistobj_iter_" + str(training_iter) + "_data_" + str(amount_data)
        mnistcost = "mnistcost_iter_" + str(training_iter) + "_data_" + str(amount_data)
        mnistacc = "mnistacc_iter_" + str(training_iter) + "_data_" + str(amount_data)

        if mnistobj and mnistcost and mnistacc in os.listdir(path):

            os.remove(path + mnistacc)
            os.remove(path + mnistcost)
            os.remove(path + mnistobj)

        train_data = [0] * amount_data
        train_labels = [0] * amount_data

        for i in range(amount_data):
            train_data[i] = np.random.uniform(size=(l_nodes[0], 1))
            train_labels[i] = np.random.uniform(size=(l_nodes[-1], 1))

        cost_testing = neu_net.train(train_data, train_labels, training_iter, learn_rate=0.01, save=True,
                                     reg_constant=0.5)

        os.remove(path + mnistacc)
        os.remove(path + mnistcost)
        os.remove(path + mnistobj)

        assert cost_testing[1][-1] <= cost_testing[1][0]

    def test_train_relu_expquadratic(self):
        l_nodes = np.random.randint(low=1, high=10, size=random.randint(5, 10))
        neu_net = NeuNetBuilder(list(l_nodes)).act("relu").cost("expquadratic").build()

        amount_data = 50
        training_iter = 100

        path = os.getcwd() + '/svc/train_objects/'
        path.replace('tests/', '')
        mnistobj = "mnistobj_iter_" + str(training_iter) + "_data_" + str(amount_data)
        mnistcost = "mnistcost_iter_" + str(training_iter) + "_data_" + str(amount_data)
        mnistacc = "mnistacc_iter_" + str(training_iter) + "_data_" + str(amount_data)

        if mnistobj and mnistcost and mnistacc in os.listdir(path):
            os.remove(path + mnistacc)
            os.remove(path + mnistcost)
            os.remove(path + mnistobj)

        train_data = [0] * amount_data
        train_labels = [0] * amount_data

        for i in range(amount_data):
            train_data[i] = np.random.uniform(size=(l_nodes[0], 1))
            train_labels[i] = np.random.uniform(size=(l_nodes[-1], 1))

        cost_testing = neu_net.train(train_data, train_labels, training_iter, learn_rate=0.01, save=True,
                                     reg_constant=0.5)

        os.remove(path + mnistacc)
        os.remove(path + mnistcost)
        os.remove(path + mnistobj)

        assert cost_testing[1][-1] <= cost_testing[1][0]