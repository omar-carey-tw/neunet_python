from svc.config import data_amount, training_iter, l_nodes, probability, distribution
from svc.config import test_data_amount, test_training_iter, test_l_nodes, test_probability, test_distribution
from helpers.helpers import generate_mask

import os
import dill as pickle

save_mask = True
mask_file = 'mask_p_' + str(probability) + '_data_' + str(data_amount) + \
            '_iter_' + str(training_iter) + '_dist_' + str(distribution)
path_to_mask = 'helpers/data_files/mask/'

if os.environ.get('TEST_FLAG'):
    mask = generate_mask(test_distribution, test_l_nodes, test_data_amount, test_training_iter, test_probability)
else:
    if mask_file not in os.listdir(path_to_mask):
        mask = generate_mask(distribution, l_nodes, data_amount, training_iter, probability)
        if save_mask:
            pickle.dump(mask, open(path_to_mask + mask_file, 'wb'))
    else:
        mask = pickle.load(open(path_to_mask + mask_file, 'rb'))
