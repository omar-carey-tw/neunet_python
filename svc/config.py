# python3 -m cProfile -s tottime svc/main.py

data_amount = 250
training_iter = 500
l_nodes = [784, 10]

probability = 1

if probability < 0.5:
    distribution = 'gaus'
else:
    distribution = 'bern'


