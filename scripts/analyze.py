import dill as pickle
from helpers.data import data

net = pickle.load(open('train_epoch_20', 'rb'))

images = data[0]
labels = data[1]
