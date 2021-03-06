from svc.net import *
from helpers.helpers import get_data

import numpy as np
import matplotlib.pyplot as plt

n = 10
min = 0.0
max = 3.0

l_nodes = [784, 10]
training_iter = 500
data_amount = 500

data = get_data(data_amount)
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
                     learn_rate=rate,
                     save=False
                     )
    co.append(cost[1][-1])
    acc.append(cost[2][-1])


fig, axs = plt.subplots(2, 1)
fig.suptitle(f"Data Amount: {data_amount} Training Iter: {training_iter}", fontsize=16)
axs[0].scatter(learn_rates, co)
axs[0].set_ylabel('Cost')
axs[0].set_xlabel('Learn Rate')

axs[1].scatter(learn_rates, acc)
axs[1].set_ylabel('Accuracy')
axs[1].set_xlabel('Learn Rate')


plt.show()
