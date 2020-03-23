# from svc.net import NeuNet
# from mnist import MNIST
import numpy as np

# PATH = "/Users/omarcarey/Desktop/aiproj/data/"
# mndata = MNIST(PATH)
# images, labels = mndata.load_training()

w = np.random.uniform(size=(3,4))
a = np.random.uniform(size=(4,1))

print(np.dot(w,a))