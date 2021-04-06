from svc.layerfactory import ReLu, Sigmoid, BuildLayerTypes, Cuadratic, Exponentialcuadratic
from numpy.testing import assert_array_equal

import numpy.linalg as la
import numpy as np


class TestActReLu:

    def test_relu_type(self):
        relu = ReLu()
        assert isinstance(relu, ReLu)

    def test_relu_act_function(self):
        relu = ReLu()
        data = [-10, 0, 10]
        process_data = np.array(data, ndmin=2).reshape((len(data), 1))

        expected_result = np.array([0, 0, 10], ndmin=2).reshape(len(data), 1)
        test_result = relu.act(process_data)

        assert (test_result == expected_result).all()

    def test_relu_dact_function(self):
        relu = ReLu()
        data = np.array([-10, 0, 10, 0.1])
        expected_result = [0, 1, 1, 1]
        test_result = relu.dactdz(data)

        assert (test_result == expected_result).all()


class TestActSigmoid:

    def test_sigmoid_type(self):
        sigmoid = Sigmoid()
        assert isinstance(sigmoid, Sigmoid)

    def test_sigmoid_act_function(self):
        sigmoid = Sigmoid()
        data = np.array([0, 0, 0])
        expected_data = [0.5, 0.5, 0.5]

        test_result = sigmoid.act(data)
        assert (test_result == expected_data).all()

    def test_sigmoid_dact_function(self):
        sigmoid = Sigmoid()
        data = np.array([0, 0, 0])
        expected_data = [0.25, 0.25, 0.25]

        test_result = sigmoid.dactdz(data)
        assert (test_result == expected_data).all()


class TestCostCuadratic:

    def test_cuadratic_type(self):
        cuadratic = Cuadratic()
        assert isinstance(cuadratic, Cuadratic)

    def test_cuadratic_cost_function(self):

        output_act = np.array([3, 5, 100])
        training_label = np.array([1, 0, 0])
        reg_const = 0.5
        weight = np.array([[1, 2, 3], [4, 5, 6]])

        cuadratic = Cuadratic()
        test_result = cuadratic.cost(output_act, training_label, weight, reg_const)

        assert test_result is not None

    def test_cuadratic_derivative_function(self):
        output_act = np.array([3, 5, 100])
        training_label = np.array([1, 0, 0])
        expected = output_act - training_label

        cuadratic = Cuadratic()
        test_result = cuadratic.dcostdact(output_act, training_label)

        assert_array_equal(test_result, expected)


class TestCostExponentialCuadratic:

    def test_exponentialcuadratic_type(self):
        exponentialcuadtraic = Exponentialcuadratic()
        assert isinstance(exponentialcuadtraic, Exponentialcuadratic)

    def test_exponentialquadratic_cost_function(self):

        output_act = np.array([.3, .5, .1])
        training_label = np.array([1, 0, 0])
        reg_const = 0.5
        weight = np.array([[1, 2, 3], [4, 5, 6]])
        expected = sum(1 / 2 * np.exp((output_act - training_label) ** 2) + reg_const / 2 * la.norm(weight, 2) ** 2)

        exponentialquadtraic = Exponentialcuadratic()
        test_result = exponentialquadtraic.cost(output_act, training_label, weight, reg_const)

        assert test_result == expected

    def test_cuadratic_derivative_function(self):
        output_act = np.array([.3, .5, .1])
        training_label = np.array([1, 0, 0])
        expected = (output_act - training_label) * np.exp((output_act - training_label) ** 2)

        exponentialquadtraic = Exponentialcuadratic()
        test_result = exponentialquadtraic.dcostdact(output_act, training_label)

        assert_array_equal(test_result, expected)


class TestBuildLayerTypes:

    def test_layer_builder_success(self):
        types = ["relu", "sigmoid"]

        layer_builder = BuildLayerTypes().build_activation_layers(types)

        assert isinstance(layer_builder[0], ReLu)
        assert isinstance(layer_builder[1], Sigmoid)
        assert isinstance(layer_builder, list)

    def test_layer_builder_fail(self):
        types = []
        layer_builder = BuildLayerTypes().build_activation_layers(types)
        assert layer_builder == []

    def test_build_weights_and_bias_success(self):
        l_nodes = [1, 2, 3, 4]
        layer_builder = BuildLayerTypes().build_weights_and_bias(l_nodes)

        assert layer_builder.get("weights") is not []
        assert len(layer_builder.get("weights")) == len(l_nodes) - 1

        assert layer_builder.get("bias") is not []
        assert len(layer_builder.get("bias")) == len(l_nodes) - 1

    def test_build_weights_and_bias_fail(self):
        l_nodes = []
        layer_builder = BuildLayerTypes().build_weights_and_bias(l_nodes)

        assert layer_builder.get("weights") == []
        assert layer_builder.get("bias") == []

    def test_build_cost_cuadratic_layer_success(self):
        cost_type = "cuadratic"

        cost_layer = BuildLayerTypes().build_cost_layer(cost_type)

        assert isinstance(cost_layer, Cuadratic)

    def test_build_cost_exponentialcuadratic_layer_success(self):
        cost_type = "exponentialcuadratic"

        cost_layer = BuildLayerTypes().build_cost_layer(cost_type)

        assert isinstance(cost_layer, Exponentialcuadratic)

    def test_build_cost_layer_fail(self):
        cost_type = ""

        cost_layer = BuildLayerTypes().build_cost_layer(cost_type)

        assert cost_layer is None