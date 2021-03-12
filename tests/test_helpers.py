from helpers.helpers import check_file, pickle_object
from svc.layernet import ROOT_DIRECTORY

import os
import pytest
import shutil


@pytest.fixture
def write_test_file():

    test_pickle_object = "This is a test"
    test_directory = "temp/test"
    test_filename = ["name", "of", "file"]

    test_file_location = os.path.join(ROOT_DIRECTORY, test_directory, "_".join(test_filename))
    os.makedirs(test_file_location.replace("_".join(test_filename), ""))

    f = open(test_file_location, "w+")

    yield test_directory, test_filename

    f.close()
    shutil.rmtree(ROOT_DIRECTORY + "/temp")


class TestHelpers:

    def test_check_file_success(self, write_test_file):

        test_directory, test_filename = write_test_file
        result = check_file(test_directory, test_filename)
        assert result

    def test_pickle_file(self):

        test_pickle_object = "This is a test"
        test_directory = os.path.join("temp", "test")
        test_filename = "name_of_file"

        test_file_location = os.path.join(ROOT_DIRECTORY, test_directory, test_filename)
        os.makedirs(test_file_location.replace(test_filename, ""))

        pickle_object(test_directory, test_filename, test_pickle_object)

        assert os.path.exists(test_file_location)

        shutil.rmtree(ROOT_DIRECTORY + "/temp")