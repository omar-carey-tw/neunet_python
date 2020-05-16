from tests.config_test import test_data_amount, test_training_iter, test_l_nodes, test_probability, test_distribution

import numpy as np


if test_distribution == 'bern':
    mask = [[[0] * len(test_l_nodes)] * test_data_amount] * test_training_iter

    for i in range(test_training_iter):
        for j in range(test_data_amount):
            for index, val in enumerate(test_l_nodes):
                mask[i][j][index] = np.random.choice([1, 0], size=(val, 1),
                                                     p=[test_probability, 1 - test_probability])
elif test_distribution == 'gaus':
    mask = [[[0] * len(test_l_nodes)] * test_data_amount] * test_training_iter

    for i in range(test_training_iter):
        for j in range(test_data_amount):
            for index, val in enumerate(test_l_nodes):
                mask[i][j][index] = np.random.randn(val, 1)
