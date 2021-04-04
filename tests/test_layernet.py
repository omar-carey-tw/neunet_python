from svc.layernet import LayerNeuNetBuilder, LayerNeuNet, ROOT_DIRECTORY

import pytest
import copy
import numpy as np
import os


# https://docs.pytest.org/en/stable/fixture.html#factories-as-fixtures


class TestLayerNeuNetBuilder:

    def test_set_layers_success(self):
        layer_types = ["relu", "sigmoid"]
        test_layerbuilder = LayerNeuNetBuilder().set_layers(layer_types)

        assert test_layerbuilder.net.layer_types is not None

    def test_set_weitghts_and_bias(self):
        l_nodes = [1, 2, 3, 4]
        test_layerbuilder = LayerNeuNetBuilder().set_weights_and_bias(l_nodes)

        assert test_layerbuilder.net.weights is not None
        assert test_layerbuilder.net.bias is not None
        assert test_layerbuilder.net.l_nodes is not None
        assert test_layerbuilder.net.layers is not None

    def test_set_cost(self):
        cost_type = "quadratic"

        test_layernetbuilder = LayerNeuNetBuilder().set_cost(cost_type)

        assert test_layernetbuilder.net.cost_layer is not None

    def test_build_success(self):
        l_nodes = [1, 2, 3, 4]
        cost_type = "quadratic"
        layer_types = ["relu"] * len(l_nodes)

        test_layernet = LayerNeuNetBuilder(). \
            set_weights_and_bias(l_nodes). \
            set_layers(layer_types). \
            set_cost(cost_type). \
            build()

        assert test_layernet.weights is not None
        assert test_layernet.bias is not None
        assert test_layernet.l_nodes is not None
        assert test_layernet.layers is not None
        assert test_layernet.layer_types is not None
        assert test_layernet.cost_layer is not None

    def test_layer_neunet_builder_fail_not_equal_lenghts(self):
        l_nodes = [1, 2, 3, 4]
        layer_types = ["relu"]

        with pytest.raises(NameError):
            LayerNeuNetBuilder().set_weights_and_bias(l_nodes).set_layers(layer_types).build()

    def test_layer_neunet_builder_fail_not_enough_layers(self):
        l_nodes = [1]
        layer_types = ["relu"]

        with pytest.raises(NameError):
            LayerNeuNetBuilder().set_weights_and_bias(l_nodes).set_layers(layer_types).build()

    def test_layernet_constructor_not_defined(self):
        with pytest.raises(NameError):
            LayerNeuNetBuilder().build()


@pytest.fixture
def create_test_layernet():
    def _test_layernet(l_nodes=[10, 7, 5, 3], layer_types="relu", cost_type="quadratic"):

        if isinstance(layer_types, list):
            layer_types_list = layer_types
        else:
            layer_types_list = [layer_types] * len(l_nodes)

        test_layernet = LayerNeuNetBuilder() \
            .set_layers(layer_types_list) \
            .set_cost(cost_type) \
            .set_weights_and_bias(l_nodes) \
            .build()

        return test_layernet

    return _test_layernet


@pytest.fixture
def get_test_data():
    def _get_test_data(data_amount, nodes_list):
        train_data = [0] * data_amount
        train_labels = [0] * data_amount

        for i in range(data_amount):
            train_data[i] = np.random.uniform(size=(nodes_list[0], 1))
            train_labels[i] = np.random.uniform(size=(nodes_list[-1], 1))

        return train_data, train_labels

    return _get_test_data


class TestLayerNeuNet:

    def test_success_init(self):
        test_layernet = LayerNeuNet()

        assert test_layernet.l_nodes is None
        assert test_layernet.layer_types is None
        assert test_layernet.layers is None
        assert test_layernet.weights is None
        assert test_layernet.bias is None
        assert test_layernet.accuracy == []
        assert test_layernet.cost == []

    def test_accuracy_true(self, create_test_layernet):
        test_layernet = create_test_layernet()
        test_output = np.array([1, 0, 0])
        test_label = np.array([1, 0, 0])

        checked_accuracy = test_layernet.check_accuracy(test_label, test_output)

        assert bool(checked_accuracy) is True

    def test_accuracy_false(self, create_test_layernet):
        test_layernet = create_test_layernet()
        test_output = np.array([0, 1, 0])
        test_label = np.array([1, 0, 0])

        checked_accuracy = test_layernet.check_accuracy(test_label, test_output)

        assert bool(checked_accuracy) is False

    def test_evaluate_success(self, create_test_layernet):
        layer_types = ["relu", "sigmoid", "sigmoid", "relu"]
        test_layernet = create_test_layernet(layer_types=layer_types)

        test_input = np.array([np.pi] * test_layernet.l_nodes[0], ndmin=2).transpose()
        test_result = test_layernet.evaluate(test_input)

        assert len(test_result) == len(test_layernet.l_nodes)
        assert len(test_result[0]) == test_layernet.l_nodes[0]
        assert len(test_result[-1]) == test_layernet.l_nodes[-1]

    def test_evaluate_fail(self, create_test_layernet):
        test_layernet = create_test_layernet()

        test_input = np.array([np.pi, np.e], ndmin=2).transpose()

        with pytest.raises(NameError):
            test_layernet.evaluate(test_input)

    def test_layernet_eval_success(self, create_test_layernet):
        test_layernet = create_test_layernet()
        test_input = np.array([np.pi] * test_layernet.l_nodes[0], ndmin=2).transpose()

        test_result = test_layernet.eval(test_input)

        assert len(test_result) == len(test_layernet.l_nodes)
        assert len(test_result[0]) == test_layernet.l_nodes[0]
        assert len(test_result[-1]) == test_layernet.l_nodes[-1]

    def test_layernet_eval_fail(self, create_test_layernet):
        test_layernet = create_test_layernet()

        test_input = np.array([np.pi, np.e], ndmin=2).transpose()

        with pytest.raises(ValueError):
            test_layernet.eval(test_input)

    def test_layernet_eval_weighted_success(self, create_test_layernet):
        test_layernet = create_test_layernet()
        test_input = np.array([np.pi] * test_layernet.l_nodes[0], ndmin=2).transpose()

        test_result = test_layernet.eval_weighted(test_input)

        assert len(test_result) == len(test_layernet.l_nodes)
        assert len(test_result[0]) == test_layernet.l_nodes[0]
        assert len(test_result[-1]) == test_layernet.l_nodes[-1]

    def test_layernet_eval_weighted_fail(self, create_test_layernet):
        test_layernet = create_test_layernet()

        test_input = np.array([np.pi, np.e], ndmin=2).transpose()

        with pytest.raises(ValueError):
            test_layernet.eval_weighted(test_input)

    @pytest.mark.flaky(max_runs=5)
    def test_train_success(self, create_test_layernet, get_test_data):
        test_layernet = create_test_layernet()
        weights_before_train = copy.deepcopy(test_layernet.weights)
        bias_before_train = copy.deepcopy(test_layernet.bias)

        data_amount = 1
        node_lengths = test_layernet.l_nodes
        training_iterations = 10

        test_train_data, test_train_labels = get_test_data(data_amount, node_lengths)

        trained_test_layernet = test_layernet.train(test_train_data, test_train_labels, training_iterations)

        assert len(trained_test_layernet.cost) == training_iterations
        assert len(trained_test_layernet.accuracy) == training_iterations

        for i in range(test_layernet.layers - 1):
            assert np.sum(weights_before_train[i]) != np.sum(trained_test_layernet.weights[i])
            assert np.sum((bias_before_train[i])) != np.sum(trained_test_layernet.bias[i])

    def test_train_save(self, create_test_layernet, get_test_data):
        test_layernet = create_test_layernet()

        data_amount = 1
        node_lengths = test_layernet.l_nodes
        training_iterations = 10
        file_name = f"mnist_obj_iter_{training_iterations}_data_{data_amount}_learning_rate_{0.5}"

        test_train_data, test_train_labels = get_test_data(data_amount, node_lengths)
        test_layernet.train(test_train_data, test_train_labels, training_iterations, save=True)
        test_trained_object = os.path.join(ROOT_DIRECTORY, "svc", "trained_objects", file_name)

        assert os.path.exists(test_trained_object)

        os.remove(test_trained_object)

    def test_train_load_previous_training_session(self, create_test_layernet, get_test_data):

        layernet_for_seeding = create_test_layernet()

        data_amount = 1
        node_lengths = layernet_for_seeding.l_nodes
        training_iterations = 10
        test_train_data, test_train_labels = get_test_data(data_amount, node_lengths)

        trained_seeded_layernet = layernet_for_seeding.train(test_train_data, test_train_labels, training_iterations,
                                                             save=True)

        trained_seeded_weights = copy.deepcopy(trained_seeded_layernet.weights)
        trained_seeded_bias = copy.deepcopy(trained_seeded_layernet.bias)

        test_layernet = create_test_layernet()
        saved_test_layernet = test_layernet.train(test_train_data, test_train_labels, training_iterations)

        assert len(saved_test_layernet.cost) == training_iterations
        assert len(saved_test_layernet.accuracy) == training_iterations

        for i in range(test_layernet.layers - 1):
            assert np.sum(trained_seeded_weights[i]) == np.sum(saved_test_layernet.weights[i])
            assert np.sum((trained_seeded_bias[i])) == np.sum(saved_test_layernet.bias[i])

        file_name = f"mnist_obj_iter_{training_iterations}_data_{data_amount}_learning_rate_{0.5}"
        test_trained_object = os.path.join(ROOT_DIRECTORY, "svc", "trained_objects", file_name)
        os.remove(test_trained_object)
