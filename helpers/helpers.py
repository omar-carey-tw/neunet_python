import numpy as np


def pickle_data(training_iter, train_data):

    dir = '/Users/omarcarey/Desktop/aiproj/NeuNet_python/'
    path_to_obj = (dir + '/svc/train_objects/').replace('tests/', '')
    meta_data = str(training_iter) + "_data_" + str(len(train_data))

    pickle_obj = "mnistobj_iter_" + meta_data
    pickle_cost = "mnistcost_iter_" + meta_data
    pickle_acc = "mnistacc_iter_" + meta_data

    return pickle_obj, pickle_cost, pickle_acc, path_to_obj


def generate_mask(distribution, l_nodes, data_amount, training_iter, probability):

    mask = [[[0] * len(l_nodes)] * data_amount] * training_iter

    if distribution == 'bern':

        for i in range(training_iter):
            for j in range(data_amount):
                for index, val in enumerate(l_nodes):
                    mask[i][j][index] = np.random.choice([1, 0], size=(val, 1),
                                                         p=[probability, 1 - probability])

    elif distribution == 'gaus':

        for i in range(training_iter):
            for j in range(data_amount):
                for index, val in enumerate(l_nodes):
                    mask[i][j][index] = np.random.randn(val, 1)

    return mask
