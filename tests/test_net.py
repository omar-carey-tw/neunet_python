import random
import pytest
import numpy as np
import os

from helpers.helpers import pickle_meta_data, generate_mask, check_previous_train
from svc.net import NeuNetBuilder

# python3 -m cProfile -s tottime svc/main.py

# todo: clean up test cleanup -> create fixture that runs after each test and checks
# that file exists after saving
# todo: make generate test data a fixture not class method


@pytest.fixture
def get_test_metadata():
    test_data_amount = 100
    test_training_iter = 100
    test_l_nodes = np.random.randint(low=1, high=10, size=random.randint(5, 10))
    return test_data_amount, test_training_iter, test_l_nodes


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

    def test_train_sigmoid(self, get_test_metadata):
        test_data_amount, test_training_iter, test_l_nodes = get_test_metadata
        neu_net = NeuNetBuilder(list(test_l_nodes)).act("sigmoid").cost("quadratic").build()

        train_data, train_labels = self.get_test_data(test_data_amount, test_l_nodes)
        test_result = neu_net.train(train_data, train_labels, test_training_iter, learn_rate=0.1, save=True)
        learn_rate = 0.1

        self.cleanup_files(test_data_amount, test_training_iter, test_learn_rate=learn_rate)
        assert test_result.get('cost')[-1] <= test_result.get('cost')[0]

    def test_train_relu(self, get_test_metadata):
        test_data_amount, test_training_iter, test_l_nodes = get_test_metadata
        neu_net = NeuNetBuilder(list(test_l_nodes)).act("relu").cost("quadratic").build()
        learn_rate = 0.01

        train_data, train_labels = self.get_test_data(test_data_amount, test_l_nodes)
        test_result = neu_net.train(train_data, train_labels, test_training_iter, learn_rate=learn_rate,
                                    save=True,
                                    reg_constant=0.5)

        self.cleanup_files(test_data_amount, test_training_iter, test_learn_rate=learn_rate)
        assert test_result.get('cost')[-1] <= test_result.get('cost')[0]

    def test_train_relu_cubic(self, get_test_metadata):
        test_data_amount, test_training_iter, test_l_nodes = get_test_metadata
        neu_net = NeuNetBuilder(list(test_l_nodes)).act("relu").cost("cubic").build()
        learn_rate = 0.01

        train_data, train_labels = self.get_test_data(test_data_amount, test_l_nodes)
        test_result = neu_net.train(train_data, train_labels, test_training_iter, learn_rate=learn_rate, save=True,
                                    reg_constant=0.5)

        self.cleanup_files(test_data_amount, test_training_iter, test_learn_rate=learn_rate)
        assert test_result.get('cost')[-1] <= test_result.get('cost')[0]

    @pytest.mark.flaky(max_runs=10)
    def test_train_relu_expquadratic(self, get_test_metadata):
        test_data_amount, test_training_iter, test_l_nodes = get_test_metadata
        neu_net = NeuNetBuilder(list(test_l_nodes)).act("relu").cost("expquadratic").build()

        train_data, train_labels = self.get_test_data(test_data_amount, test_l_nodes)
        learn_rate = 0.01

        test_result = neu_net.train(train_data, train_labels, test_training_iter, learn_rate=learn_rate, save=True,
                                    reg_constant=0.5)

        self.cleanup_files(test_data_amount, test_training_iter, test_learn_rate=learn_rate)
        assert test_result.get('cost')[-1] <= test_result.get('cost')[0]

    @pytest.mark.flaky(max_runs=10)
    def test_train_relu_expquadratic_dropout(self, get_test_metadata):
        test_data_amount, test_training_iter, test_l_nodes = get_test_metadata
        neu_net = NeuNetBuilder(list(test_l_nodes)).act("relu").cost("expquadratic").build()

        train_data, train_labels = self.get_test_data(test_data_amount, test_l_nodes)
        probability = 0.8
        learn_rate = 0.01

        test_result = neu_net.train(train_data, train_labels, test_training_iter, probability=probability,
                                    learn_rate=learn_rate, save=True)

        self.cleanup_files(test_data_amount, test_training_iter, test_learn_rate=learn_rate, test_probability=probability)
        assert test_result.get('cost')[-1] <= test_result.get('cost')[0]

    def cleanup_files(self, test_data_amount, test_training_iter, test_probability=None, test_learn_rate=None):

        pickle_obj, pickle_cost, pickle_acc, path_to_obj = pickle_meta_data(test_training_iter, test_data_amount, test_probability, test_learn_rate)

        path = (os.getcwd() + '/svc/train_objects/').replace('tests/', '')

        os.remove(path + pickle_obj)
        os.remove(path + pickle_cost)
        os.remove(path + pickle_acc)

    def get_test_data(self, test_data_amount, test_l_nodes):
        train_data = [0] * test_data_amount
        train_labels = [0] * test_data_amount

        for i in range(test_data_amount):
            train_data[i] = np.random.uniform(size=(test_l_nodes[0], 1))
            train_labels[i] = np.random.uniform(size=(test_l_nodes[-1], 1))

        return train_data, train_labels


class TestHelpers:

    def test_pickle_meta_data(self, get_test_metadata):
        test_data_amount, test_training_iter, _ = get_test_metadata
        probability = 0.7
        learning_rate = .5
        EXPECTED_OBJ_NAME = f"mnist_obj_iter_{test_training_iter}_data_{test_data_amount}_prob_{probability}" \
                            f"_learn_rate_{learning_rate}"
        EXPECTED_COST_NAME = f"mnist_cost_iter_{test_training_iter}_data_{test_data_amount}_prob_{probability}" \
                            f"_learn_rate_{learning_rate}"
        EXPECTED_ACC_NAME = f"mnist_acc_iter_{test_training_iter}_data_{test_data_amount}_prob_{probability}" \
                            f"_learn_rate_{learning_rate}"
        EXPECTED_PATH_TO_OBJ = '/Users/omarcarey/Desktop/aiproj/NeuNet_python/svc/train_objects/'

        test_pickle_obj, test_pickle_cost, test_pickle_acc, test_path_to_obj = pickle_meta_data(test_training_iter,
                                                                                                test_data_amount,
                                                                                                probability=probability,
                                                                                                learn_rate=learning_rate)
        assert EXPECTED_OBJ_NAME == test_pickle_obj
        assert EXPECTED_COST_NAME == test_pickle_cost
        assert EXPECTED_ACC_NAME == test_pickle_acc
        assert EXPECTED_PATH_TO_OBJ == test_path_to_obj

    def test_generate_mask_no_dist(self, get_test_metadata):

        test_data_amount, test_training_iter, test_l_nodes = get_test_metadata
        test_mask = generate_mask(test_l_nodes, test_data_amount, test_training_iter, probability=None)

        assert len(test_mask) == test_training_iter
        assert len(test_mask[0]) == test_data_amount
        assert len(test_mask[0][0]) == len(test_l_nodes)

        for i in range(len(test_l_nodes)):
            assert test_mask[0][0][i] == 1

    def test_generate_mask_bern_dist(self, get_test_metadata):

        test_data_amount, test_training_iter, test_l_nodes = get_test_metadata
        test_mask = generate_mask(test_l_nodes, test_data_amount, test_training_iter, probability=0.8)

        assert len(test_mask) == test_training_iter
        assert len(test_mask[0]) == test_data_amount
        assert len(test_mask[0][0]) == len(test_l_nodes)

        for i in range(len(test_l_nodes)):
            assert len(test_mask[0][0][i]) == test_l_nodes[i]

    def test_generate_mask_gaus_dist(self, get_test_metadata):

        test_data_amount, test_training_iter, test_l_nodes = get_test_metadata
        test_mask = generate_mask(test_l_nodes, test_data_amount, test_training_iter, probability=0.1)

        assert len(test_mask) == test_training_iter
        assert len(test_mask[0]) == test_data_amount
        assert len(test_mask[0][0]) == len(test_l_nodes)

        for i in range(len(test_l_nodes)):
            assert len(test_mask[0][0][i]) == test_l_nodes[i]

    def test_check_previous_train_no_previous_run(self, get_test_metadata):
        test_data_amount, test_training_iter, _ = get_test_metadata

        check = check_previous_train(test_training_iter, test_data_amount, probability=.5, learn_rate=.5)

        assert check is None

    @pytest.mark.skip
    def test_check_previous_train_yes_previous_run(self, get_test_metadata):
        TEST_DATA_AMOUNT = 38
        TEST_TRAIN_ITER = 321
        PROB = .5
        LEARN_RATE = .7

        check = check_previous_train(TEST_TRAIN_ITER, TEST_DATA_AMOUNT, PROB, LEARN_RATE)

        assert check is not None


class TestLayer:

    def test_layer_eval(self):
        












