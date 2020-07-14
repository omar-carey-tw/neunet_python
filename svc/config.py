
data_amount = 100
training_iter = 100
l_nodes = [784, 10]
probability = 1

if probability < 0.5:
    distribution = 'gaus'
else:
    distribution = 'bern'


