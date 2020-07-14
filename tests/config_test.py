import numpy as np
import random

test_data_amount = 50
test_training_iter = 100
test_l_nodes = np.random.randint(low=1, high=10, size=random.randint(5, 10))
test_probability = 0.5

if test_probability < 0.5:
    test_distribution = 'gaus'
else:
    test_distribution = 'bern'
