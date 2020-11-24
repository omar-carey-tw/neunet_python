from helpers.helpers import get_data
from svc.net import *
import matplotlib.pyplot as plt

l_nodes = [784, 10]
training_iter = 150
data_amount = 250
batch_size = 100

learn_rate = 0.9
data = get_data(data_amount)
images = data[0]
labels = data[1]

accuracy = []

#todo: add mean and std to plot
for i in range(batch_size):

    neu_net = NeuNetBuilder(l_nodes).act("relu").cost("expquadratic").build()
    result = neu_net.train(images,
                           labels,
                           training_iter,
                           learn_rate=learn_rate
                           )
    acc = float(result.get('accuracy')[-1])
    accuracy.append(acc)

plt.hist(accuracy)
plt.xlabel('Accuracy')
plt.ylabel('Count')
plt.title(f"Distributions of {batch_size} batch sizes after {training_iter} training iterations \n"
          f"Batch Size: {batch_size} \n "
          f"Data Amount: {data_amount} \n "
          f"Training Iter: {training_iter}")

plt.show()