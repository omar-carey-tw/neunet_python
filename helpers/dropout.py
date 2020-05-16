from svc.config import data_amount, training_iter, l_nodes, probability, distribution

import os
import numpy as np
import dill as pickle

save_mask = True
mask_file = 'mask_p_' + str(probability) + '_data_' + str(data_amount) + \
            '_iter_' + str(training_iter) + '_dist_' + str(distribution)
path_to_mask = 'helpers/data_files/mask/'

if mask_file not in os.listdir(path_to_mask):
    mask = [[[0] * len(l_nodes)] * data_amount] * training_iter

    if distribution == 'bern':

        for i in range(training_iter):
            for j in range(data_amount):
                for index, val in enumerate(l_nodes):
                    mask[i][j][index] = np.random.choice([1, 0], size=(val, 1),
                                                         p=[probability, 1-probability])

    elif distribution == 'gaus':

        for i in range(training_iter):
            for j in range(data_amount):
                for index, val in enumerate(l_nodes):
                    mask[i][j][index] = np.random.randn(val, 1)

    if save_mask:
        pickle.dump(mask, open(path_to_mask + mask_file, 'wb'))
else:
    mask = pickle.load(open(path_to_mask + mask_file, 'rb'))
