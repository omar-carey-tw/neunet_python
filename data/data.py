from mnist import MNIST
from svc.config import data_amount

import numpy as np
import dill as pickle
import os


data_file = 'data_amount_' + str(data_amount)
path_to_data = ''

if data_file not in os.listdir():

    PATH = "/Users/omarcarey/Desktop/aiproj/data/"
    mndata = MNIST(PATH)

    images, labels = mndata.load_training_in_batches(data_amount)

    proc_labels = np.zeros(shape=(len(labels), 10, 1))
    proc_images = np.zeros(shape=(len(images), len(images[0]), 1))

    gray_scale = 255

    for index, val in enumerate(labels):
        proc_labels[index][val] = 1
        proc_images[index] = np.array(images[index]).reshape(len(images[index]), 1) / gray_scale

    data = (proc_images, proc_labels)
    pickle.dump(data, open(data_file, 'wb'))

else:

    data = pickle.load(open(data_file, 'rb'))
