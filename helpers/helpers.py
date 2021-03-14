import numpy as np
import os
import dill as pickle

from mnist import MNIST
from typing import List

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__)).replace('helpers', '')


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

            if os.getenv('SAVE_MASK'):
                print(f"Saving mask: {mask_file}", "\n")
                pickle.dump(mask, open(path_to_mask + mask_file, 'wb'))

        else:
            print(f"Loading saved mask: {mask_file}", "\n ")
            mask = pickle.load(open(path_to_mask + mask_file, 'rb'))

    return mask


def get_data(data_amount, save_data=True):

    data_file_list = ["data_amount", f"{str(data_amount)}"]
    directory = os.path.join("helpers", "data_files", "data")
    checked_file = check_file(directory, data_file_list)
    data = {}

    if checked_file:

        # data = pickle.load(open(path_to_data + data_file, 'rb'))
        pass

    else:

        path = "/Users/omarcarey/Desktop/aiproj/data/"
        mndata = MNIST(path)

        images, labels = mndata.load_training_in_batches(data_amount)

        processed_labels = np.zeros(shape=(len(labels), 10, 1))
        processed_images = np.zeros(shape=(len(images), len(images[0]), 1))

        gray_scale = 255

        for index, val in enumerate(labels):
            processed_labels[index][val] = 1
            processed_images[index] = np.array(images[index]).reshape(len(images[index]), 1) / gray_scale

        data["images"] = processed_images
        data["labels"] = processed_labels

        if save_data:
            pickle_object(directory, data_file_list, data)

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


def check_file(directory, file_name_list: List):

    file = os.path.join(ROOT_DIRECTORY, directory, "_".join(file_name_list))

    if os.path.exists(file):
        return True

    return False


def pickle_object(directory, file_name_list, object):

    file_location = os.path.join(ROOT_DIRECTORY, directory, "_".join(file_name_list))
    pickle.dump(object, open(file_location, 'wb'))


# def check_previous_train(training_iter, data_amount, probability, learn_rate):
#     result = None
#
#     pickle_obj, pickle_cost, pickle_acc, path_to_obj = pickle_meta_data(training_iter, data_amount, probability,
#                                                                         learn_rate)
#
#     if pickle_obj and pickle_cost and pickle_acc in os.listdir(path_to_obj):
#         print(f"Loading trained Net: {pickle_obj}", "\n ----------")
#         neunet = pickle.load(open(path_to_obj + pickle_obj, 'rb'))
#         cost = pickle.load(open(path_to_obj + pickle_cost, 'rb'))
#         acc = pickle.load(open(path_to_obj + pickle_acc, 'rb'))
#
#         result = {
#             "net": neunet,
#             "cost": cost,
#             "accuracy": acc
#         }
#
#     return result

# def pickle_meta_data(training_iter, train_data_amount, probability=None, learn_rate=None):
#
#     dir = os.path.dirname(os.path.abspath(__file__))
#     path_to_obj = (dir + '/svc/trained_objects/').replace('tests/', '')
#     meta_data = str(training_iter) + \
#                 "_data_" + str(train_data_amount) + \
#                 "_prob_" + str(probability) + \
#                 "_learn_rate_" + str(learn_rate)
#
#     pickle_obj = "mnist_obj_iter_" + meta_data
#     pickle_cost = "mnist_cost_iter_" + meta_data
#     pickle_acc = "mnist_acc_iter_" + meta_data
#
#     return pickle_obj, pickle_cost, pickle_acc, path_to_obj