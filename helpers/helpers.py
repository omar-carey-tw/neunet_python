import numpy as np
import os
import dill as pickle

from typing import List

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__)).replace('helpers', '')


def generate_mask(l_nodes, data_amount, training_iter, probability, save=False):
    mask = [[[1] * len(l_nodes)] * data_amount] * training_iter

    if probability is not None:

        distribution = determine_dist(probability)
        direcotry = os.path.join("helpers", "data_files", "mask")
        file_name_list = ["probability", f"{probability}", "data_amount", f"{data_amount}", "training_iterations",
                          f"{training_iter}", "l_nodes", f"{l_nodes}"]

        checked_file = check_file(direcotry, file_name_list)

        if not checked_file:
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

            if save:
                print(f"Saving mask: {'_'.join(file_name_list)}", "\n")
                pickle_object(direcotry, file_name_list, mask)
        else:
            print(f"Loading saved mask: {'_'.join(file_name_list)}", "\n ")
            mask = pickle_load(direcotry, file_name_list)

    return mask


def get_data(data_amount, save_data=True):

    directory = os.path.join("helpers", "data_files", "data")
    file_name_list = ["data_amount", f"{data_amount}"]

    checked_file = check_file(directory, file_name_list)

    if checked_file:
        file_name = "_".join(file_name_list)
        path_to_data = os.path.join(ROOT_DIRECTORY, directory, file_name)

        data = pickle.load(open(path_to_data, 'rb'))

    else:
        path_to_full_set = os.path.join(ROOT_DIRECTORY, "mnistdataset", "FULL_SET")
        data = pickle.load(open(path_to_full_set, 'rb'))

        images, labels = data.get("images"), data.get("labels")

        processed_labels = np.zeros(shape=(data_amount, 10, 1))
        processed_images = np.zeros(shape=(data_amount, len(images[0]), 1))

        gray_scale = 255

        for i in range(data_amount):
            processed_labels[i][labels[i]] = 1
            processed_images[i] = np.array(images[i]).reshape(len(images[i]), 1) / gray_scale

        processed_data = {
            "images": processed_images,
            "labels": processed_labels
        }

        data = processed_data

    if save_data:
        file_name_list = ["data_amount", f"{data_amount}"]
        directory = os.path.join("helpers", "data_files", "data")
        pickle_object(directory, file_name_list, data)

    return data


def determine_dist(probability):
    if probability < 0.5:
        distribution = 'gaus'
    else:
        distribution = 'bern'

    return distribution


def check_file(directory, file_name_list: List):

    file_name = "_".join(file_name_list)

    file = os.path.join(ROOT_DIRECTORY, directory, file_name)

    if os.path.exists(file):
        return True

    return False


def pickle_object(directory, file_name_list, object):

    file_name = "_".join(file_name_list)
    file_location = os.path.join(ROOT_DIRECTORY, directory, file_name)
    pickle.dump(object, open(file_location, 'wb+'))


def pickle_load(directory, file_name_list):
    file_name = "_".join(file_name_list)
    file_location = os.path.join(ROOT_DIRECTORY, directory, file_name)
    unpickled_file = pickle.load(open(file_location, 'rb'))

    return unpickled_file

