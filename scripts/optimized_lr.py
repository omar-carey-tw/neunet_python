from svc.net import *
from helpers.data import data

import numpy as np
import matplotlib.pyplot as plt

n = 25
min = 0.0
max = 3.5

l_nodes = [784, 10]
training_iter = 250

images = data[0]
labels = data[1]

learn_rates = np.linspace(min, max, n)

acc = []
co = []

for rate in learn_rates:
    print(rate)
    net = NeuNetBuilder(l_nodes).act("relu").cost("expquadratic").build()
    cost = net.train(images,
                     labels,
                     training_iter,
                     rate
                     )
    co.append(cost[1][-1])
    acc.append(cost[2][-1])

print(learn_rates)

fig, axs = plt.subplots(2, 1)
axs[0].scatter(learn_rates, co)
axs[0].set_ylabel('Cost')

axs[1].scatter(learn_rates, acc)
axs[1].set_ylabel('Accuracy')

plt.show()
