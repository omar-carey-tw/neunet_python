from mnist import MNIST
import numpy as np

PATH = "/Users/omarcarey/Desktop/aiproj/data/"
mndata = MNIST(PATH)
images, labels = mndata.load_training()

proc_labels = np.zeros(shape=(len(labels), 10, 1))
proc_images = np.zeros(shape=(len(images), len(images[0]), 1))

gray_scale = 255

for index, val in enumerate(labels):
    proc_labels[index][val] = 1
    proc_images[index] = np.array(images[index]).reshape(len(images[index]), 1) / gray_scale

data = (proc_images, proc_labels)
