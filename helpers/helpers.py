import numpy as np
import os
import dill as pickle


def pickle_data(training_iter, train_data_amount, probability=None, learn_rate=None):

    dir = '/Users/omarcarey/Desktop/aiproj/NeuNet_python/'
    path_to_obj = (dir + '/svc/train_objects/').replace('tests/', '')
    meta_data = str(training_iter) + \
                "_data_" + str(train_data_amount) + \
                "_prob_" + str(probability) + \
                "_learn_rate_" + str(learn_rate)

    pickle_obj = "mnistobj_iter_" + meta_data
    pickle_cost = "mnistcost_iter_" + meta_data
    pickle_acc = "mnistacc_iter_" + meta_data

    return pickle_obj, pickle_cost, pickle_acc, path_to_obj


def generate_mask(l_nodes, data_amount, training_iter, probability):

    mask = [[[1] * len(l_nodes)] * data_amount] * training_iter

    if probability is not None:

        distribution = determine_dist(probability)
        mask_file, path_to_mask = mask_metadata(data_amount, training_iter, probability, distribution)

        if mask_file not in os.listdir(path_to_mask):
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

            if os.getenv('SAVE_MASK') and not os.getenv('TEST_FLAG'):
                print(f"Saving mask: {mask_file}", "\n")
                pickle.dump(mask, open(path_to_mask + mask_file, 'wb'))

        else:
            print(f"Loading saved mask: {mask_file}", "\n ")
            mask = pickle.load(open(path_to_mask + mask_file, 'rb'))

    return mask


def get_data(data_amount, save_data=True):
    from mnist import MNIST

    data_file = 'data_amount_' + str(data_amount)
    path_to_data = '/Users/omarcarey/Desktop/aiproj/NeuNet_python/helpers/data_files/data/'

    if data_file not in os.listdir(path_to_data):

        path = "/Users/omarcarey/Desktop/aiproj/data/"
        mndata = MNIST(path)

        images, labels = mndata.load_training_in_batches(data_amount)

        proc_labels = np.zeros(shape=(len(labels), 10, 1))
        proc_images = np.zeros(shape=(len(images), len(images[0]), 1))

        gray_scale = 255

        for index, val in enumerate(labels):
            proc_labels[index][val] = 1
            proc_images[index] = np.array(images[index]).reshape(len(images[index]), 1) / gray_scale

        data = (proc_images, proc_labels)
        if save_data:
            pickle.dump(data, open(path_to_data + data_file, 'wb'))

    else:

        data = pickle.load(open(path_to_data + data_file, 'rb'))

    return data


def determine_dist(probability):
    if probability < 0.5:
        distribution = 'gaus'
    else:
        distribution = 'bern'

    return distribution


def mask_metadata(data_amount, training_iter, probability, distribution):
    mask_file = 'mask_p_' + str(probability) + '_data_' + str(data_amount) + \
                '_iter_' + str(training_iter) + '_dist_' + str(distribution)
    dir = '/Users/omarcarey/Desktop/aiproj/NeuNet_python/'

    path_to_mask = (dir + 'helpers/data_files/mask/').replace('tests/', '')

    return mask_file, path_to_mask
