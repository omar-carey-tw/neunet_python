import os

def pickle_data(training_iter, train_data):

    dir = '/Users/omarcarey/Desktop/aiproj/NeuNet_python/'
    path_to_obj = (dir + '/svc/train_objects/').replace('tests/', '')
    meta_data = str(training_iter) + "_data_" + str(len(train_data))

    pickle_obj = "mnistobj_iter_" + meta_data
    pickle_cost = "mnistcost_iter_" + meta_data
    pickle_acc = "mnistacc_iter_" + meta_data

    return pickle_obj, pickle_cost, pickle_acc, path_to_obj


