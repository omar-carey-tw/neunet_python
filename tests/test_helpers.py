from helpers.helpers import check_file, pickle_object, get_data, generate_mask
from svc.layernet import ROOT_DIRECTORY

import os
import pytest
import shutil


@pytest.fixture
def write_test_file():

    test_directory = "temp/test"
    test_filename = ["name", "of", "file"]

    test_file_location = os.path.join(ROOT_DIRECTORY, test_directory, "_".join(test_filename))
    os.makedirs(test_file_location.replace("_".join(test_filename), ""))

    f = open(test_file_location, "w+")

    yield test_directory, test_filename

    f.close()


@pytest.fixture(autouse=True)
def clean_up():
    yield

    path_to_temp = os.path.join(ROOT_DIRECTORY, "temp")

    if os.path.exists(path_to_temp):
        shutil.rmtree(ROOT_DIRECTORY + "/temp")


class TestHelpers:

    def test_check_file_success(self, write_test_file):

        test_directory, test_filename = write_test_file
        result = check_file(test_directory, test_filename)
        assert result

    def test_pickle_file(self):

        test_pickle_object = "This is a test"
        test_directory = os.path.join("temp", "test")
        test_filename_list = ["name", "of", "file"]

        test_filename = "_".join(test_filename_list)

        test_file_location = os.path.join(ROOT_DIRECTORY, test_directory, test_filename)
        os.makedirs(test_file_location.replace(test_filename, ""))

        pickle_object(test_directory, test_filename_list, test_pickle_object)

        assert os.path.exists(test_file_location)

    def test_get_unsaved_data_no_storage(self):

        test_data_amount = 3
        test_data = get_data(test_data_amount, save_data=False)

        assert len(test_data.get("images")) == test_data_amount
        assert len(test_data.get("labels")) == test_data_amount

    def test_get_unsaved_data_with_storage(self):

        test_data_amount = 3
        test_data = get_data(test_data_amount, save_data=True)

        directory = os.path.join("helpers", "data_files", "data")
        file_name_list = ["data_amount", f"{test_data_amount}"]
        test_file_name = "_".join(file_name_list)
        test_file_path = os.path.join(ROOT_DIRECTORY, directory, test_file_name)

        assert len(test_data.get("images")) == test_data_amount
        assert len(test_data.get("labels")) == test_data_amount
        assert os.path.exists(test_file_path)

        os.remove(test_file_path)

    def test_get_saved_data(self):

        test_data_amount = 3
        test_object = {
            "images": [0]*test_data_amount,
            "labels": [0]*test_data_amount
        }

        directory = os.path.join("helpers", "data_files", "data")
        file_name_list = ["data_amount", f"{test_data_amount}"]
        test_file_name = "_".join(file_name_list)
        test_file_path = os.path.join(ROOT_DIRECTORY, directory, test_file_name)

        pickle_object(directory, file_name_list, test_object)

        test_data = get_data(test_data_amount, save_data=False)

        assert len(test_data.get("images")) == test_data_amount
        assert len(test_data.get("labels")) == test_data_amount

        os.remove(test_file_path)

    @pytest.mark.flaky(max_runs=10)
    def test_generate_mask(self):

        test_l_nodes = [10, 7, 5, 3]
        test_data_amount = 100
        test_training_iterations = 100
        probability = 0.5

        test_mask = generate_mask(test_l_nodes, test_data_amount, test_training_iterations, probability)

        assert len(test_mask) == test_training_iterations
        assert len(test_mask[0]) == test_data_amount
        assert len(test_mask[0][0]) == len(test_l_nodes)
        assert test_mask[0][0][0][0] != 1