from helpers.helpers import get_data
from svc.net import *
import matplotlib.pyplot as plt

l_nodes = [784, 10]
training_iter = 500
data_amount = 500

data = get_data(data_amount)
images = data[0]
labels = data[1]

probabilities = [0, 0.5, 0.7, 0.9, 1]
learn_rate = 0.9
save_obj = True

fig, axs = plt.subplots(len(probabilities), 2, sharex=True)

for index, probability in enumerate(probabilities):
    neu_net = NeuNetBuilder(l_nodes).act("relu").cost("expquadratic").build()
    cost = neu_net.train(images,
                         labels,
                         training_iter,
                         learn_rate=learn_rate,
                         probability=probability,
                         save=save_obj
                         )
    axs[index, 0].plot(list(range(training_iter)), cost[1])
    axs[index, 0].set_title(f'Cost w/ prob {probability}, LR {learn_rate}')

    axs[index, 1].plot(list(range(training_iter)), cost[2])
    axs[index, 1].set_title(f'Acc w/ prob {probability}, LR {learn_rate}')

plt.show()
