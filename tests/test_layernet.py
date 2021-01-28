from svc.layernet import LayerNeuNetBuilder, LayerNeuNet

import pytest
import numpy as np

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
        layer_types = ["relu"]*len(l_nodes)

        test_layernet = LayerNeuNetBuilder().\
            set_weights_and_bias(l_nodes).\
            set_layers(layer_types).\
            set_cost(cost_type).\
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


@pytest.fixture()
def successful_layernet_init():
    l_nodes = [10, 7, 5, 3]
    layer_types = ["relu"] * len(l_nodes)
    cost_type = "quadratic"

    test_layernet = LayerNeuNetBuilder(). \
        set_layers(layer_types). \
        set_cost(cost_type). \
        set_weights_and_bias(l_nodes). \
        build()

    return test_layernet


class TestLayerNeuNet:

    def test_success_init(self):

        test_layernet = LayerNeuNet()

        assert test_layernet.l_nodes is None
        assert test_layernet.layer_types is None
        assert test_layernet.layers is None
        assert test_layernet.weights is None
        assert test_layernet.bias is None

    def test_evaluate_success(self, successful_layernet_init):

        test_layernet = successful_layernet_init

        test_input = np.array([np.pi]*test_layernet.l_nodes[0], ndmin=2).transpose()
        test_result = test_layernet.evaluate(test_input)

        assert len(test_result) == len(test_layernet.l_nodes)
        assert len(test_result[0]) == test_layernet.l_nodes[0]
        assert len(test_result[-1]) == test_layernet.l_nodes[-1]

    def test_evaluate_fail(self, successful_layernet_init):

        test_layernet = successful_layernet_init

        test_input = np.array([np.pi, np.e], ndmin=2).transpose()

        with pytest.raises(NameError):
            test_layernet.evaluate(test_input)

    def test_train_success(self):

        #todo: change pytest fixture to take in param to make custom layernet
        assert 0 is 1
