# from svc.net import NeuNet
# from mnist import MNIST
import numpy as np

# PATH = "/Users/omarcarey/Desktop/aiproj/data/"
# mndata = MNIST(PATH)
# images, labels = mndata.load_training()

def act(y):
        # sigmoid
        return 1/(1+np.exp(-y))

w = np.random.uniform(size=(4,1))
a = np.random.uniform(size=(4,1))

# print(w)
# print(w**2)
# print(act(w))

for i in range(10,1,-1):
    print(i)
