from main import training_iter, l_nodes, probability
from svc.config import data_amount

import numpy as np

mask = [[]]*training_iter

for i in range(training_iter):
    for j in range(data_amount):
        for k in range(l_nodes):
            mask[i][j] = np.random.choice([1, 0], size=(k, 1), p=[probability, 1-probability]).astype(np.bool)



