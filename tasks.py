from invoke import task
import os

from helpers.helpers import ROOT_DIRECTORY, pickle_object
from mnist import MNIST

PICKLED_MNIST_DATA_DIRECTORY = "mnistdataset"


@task
def download_and_pickle_mnist_data(ctx, directory=PICKLED_MNIST_DATA_DIRECTORY):

    data_file_list = ["FULL", "SET"]
    path = os.path.join(ROOT_DIRECTORY, "mnistdataset")
    mndata = MNIST(path)
    images, labels = mndata.load_training()

    data = {
        "images": images,
        "labels": labels
    }

    pickle_object(directory, data_file_list, data)

