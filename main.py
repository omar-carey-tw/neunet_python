from svc.net import *
from typing import *

import numpy as np
import matplotlib.pyplot as plt
from svc.data import data


l_nodes = [784,15,10]
neu_net = NeuNet(l_nodes)
old_weights = neu_net.weights
old_bias = neu_net.bias

training_iter = 100
images = data[0]
labels = data[1]

cost = neu_net.train(images, labels, training_iter)

new_weights = neu_net.weights
new_bias = neu_net.bias

plt.plot(list(range(training_iter)), cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

for i in range(len(new_weights)):
    print(old_weights[i])
    print(new_weights[i])
    print("\n")

print(cost[0],cost[-1])