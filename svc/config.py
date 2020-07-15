import numpy as np
import random

data_amount = 100
training_iter = 100
l_nodes = [784, 10]
probability = 1

if probability < 0.5:
    distribution = 'gaus'
else:
    distribution = 'bern'

# Test Data
test_data_amount = 100
test_training_iter = 100
test_l_nodes = np.random.randint(low=1, high=10, size=random.randint(5, 10))
test_probability = 0.8

if test_probability < 0.5:
    test_distribution = 'gaus'
else:
    test_distribution = 'bern'