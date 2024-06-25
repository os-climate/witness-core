'''
Copyright 2022 Airbus SAS
Modifications on {} Copyright 2024 Capgemini
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

from sostrades_core.sos_processes.script_test_all_usecases import _test_all_usecases


class TestUseCases(unittest.TestCase):
    """
    Usecases test class
    """

    def setUp(self):
        """Initialize third data needed for testing"""
        self.pp = pprint.PrettyPrinter(indent=4, compact=True)
        self.processes_repo = 'climateeconomics.sos_processes'
        self.maxDiff = None

    def test_all_usecases(self):
        test_passed, output_error = _test_all_usecases(processes_repo=self.processes_repo)
        if not test_passed:
            raise Exception(f'{output_error}')
