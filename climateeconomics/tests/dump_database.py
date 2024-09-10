'''
Copyright 2022 Airbus SAS
Modifications on 27/11/2023 Copyright 2023 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import pprint
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal

from climateeconomics.tests.dump_database_configure import (
    dump_json_for_database_energy,
    dump_value_into_dict,
    retrieve_input_from_dict,
)


class TestUseCases(unittest.TestCase):
    """
    Usecases test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.pp = pprint.PrettyPrinter(indent=4, compact=True)
        self.processes_repo = 'climateeconomics.sos_processes.iam.witness.witness_dev'
        self.maxDiff = None

    def test_01_dump_and_retrieve_array(self):
        array_1 = np.array([1, 7, 8, 10])
        array_1_dict = dump_value_into_dict(array_1, 'array')
        array_1_retrieved = retrieve_input_from_dict(array_1_dict, 'array')
        assert_array_equal(array_1, array_1_retrieved)

    def test_02_dump_and_retrieve_dataframe(self):
        dataframe_1 = pd.DataFrame([[1, 2], [3, 4]], columns=['column1', 'column2'], index=['row1', 'row2'])
        dataframe_1_dict = dump_value_into_dict(dataframe_1, 'dataframe')
        dataframe_1_retrieved = retrieve_input_from_dict(dataframe_1_dict, 'dataframe')
        assert_frame_equal(dataframe_1, dataframe_1_retrieved)

    def test_03_dump_usecases_data_into_json_file(self):
        dump_json_for_database_energy(self.processes_repo)
