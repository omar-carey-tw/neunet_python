from svc.config import data_amount, training_iter, l_nodes, probability

import os
import numpy as np
import dill as pickle

mask_file = 'mask_p_' + str(probability) + '_data_' + str(data_amount) + '_iter_' + str(training_iter)
path_to_mask = 'helpers/data_files/'
save = True

if mask_file not in os.listdir(path_to_mask):

    mask = {
        "nodes": [[[0]*len(l_nodes)]*data_amount]*training_iter,
        "prob": probability
           }

    for i in range(training_iter):
        for j in range(data_amount):
            for index, val in enumerate(l_nodes):
                mask.get("nodes")[i][j][index] = np.random.choice([1, 0], size=(val, 1), p=[probability, 1-probability]).astype(np.bool)

    if save:
        pickle.dump(mask, open(path_to_mask + mask_file, 'wb'))

else:
    mask = pickle.load(open(path_to_mask + mask_file), 'rb')
