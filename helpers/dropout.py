from svc.config import data_amount, training_iter, l_nodes, probability
from tests.test_config import test_data_amount, test_training_iter, test_l_nodes, test_probability

import os
import numpy as np
import dill as pickle

mask_file = 'mask_p_' + str(probability) + '_data_' + str(data_amount) + '_iter_' + str(training_iter)

path_to_mask = 'helpers/data_files/mask/'
save_mask = True

if os.environ.get('TEST_FLAG'):
    mask = [[[0] * len(test_l_nodes)] * test_data_amount] * test_training_iter

    for i in range(test_training_iter):
        for j in range(test_data_amount):
            for index, val in enumerate(test_l_nodes):
                mask[i][j][index] = np.random.choice([1, 0], size=(val, 1),
                                                     p=[test_probability, 1 - test_probability])

elif mask_file not in os.listdir(path_to_mask):

    mask = [[[0]*len(l_nodes)]*data_amount]*training_iter

    for i in range(training_iter):
        for j in range(data_amount):
            for index, val in enumerate(l_nodes):
                mask[i][j][index] = np.random.choice([1, 0], size=(val, 1),
                                                     p=[probability, 1-probability]).astype(np.bool)

    if save_mask:
        pickle.dump(mask, open(path_to_mask + mask_file, 'wb'))

else:
    mask = pickle.load(open(path_to_mask + mask_file, 'rb'))
