import dill as pickle
from helpers.data import data

dir = '/Users/omarcarey/Desktop/aiproj/NeuNet_python/'
file = 'train_epoch_20'
net = pickle.load(open(dir + file, 'rb'))

images = data[0]
labels = data[1]
