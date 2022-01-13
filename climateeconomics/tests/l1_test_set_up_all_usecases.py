'''
Copyright 2022 Airbus SAS

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
from numpy import empty
from array import array
from sos_trades_core.study_manager.study_manager import StudyManager


'''
Created on 19 Oct 2021

@author: NG8B00C
'''

from sos_trades_core.sos_processes.processes_factory import SoSProcessFactory
from importlib import import_module
from os.path import dirname, isdir
from os import listdir, makedirs
from tempfile import gettempdir
from collections.abc import MutableMapping
from contextlib import suppress
from sos_trades_core.execution_engine.namespace import Namespace
import unittest


class ArrayInDspaceTest(unittest.TestCase):
    """Exception if a usecase fails.

    Attributes:
        message -- explanation of the error
    """
    error_list = []

    def __init__(self, error_list):
        self.error_list = error_list
        super().__init__(self.error_list)

    def __str__(self):
        return '\n' + '\n'.join(self.error_list)

    def test_array_in_dspace(self, processes_repo='climateeconomics.sos_processes'):
        '''
        set all usecases of a specific repository
        Raise an exception if a array is in dspace
        '''
        key_to_check_list = ['value', 'upper_bnd',
                             'lower_bnd', 'activated_elem']
        # Retrieve all processes for this repository only
        process_factory = SoSProcessFactory(additional_repository_list=[
                                            processes_repo], search_python_path=False)
        process_dict = process_factory.get_processes_dict()
        # Set dir to dump reference
        dump_dir = f'{ gettempdir() }/references'
        if not isdir(dump_dir):
            makedirs(dump_dir)
        for repository in process_dict:
            for process in process_dict[repository]:

                imported_module = import_module(
                    '.'.join([repository, process]))

                if imported_module is not None and imported_module.__file__ is not None:
                    process_directory = dirname(imported_module.__file__)
                    # Set up all usecases
                    for usecase_py in listdir(process_directory):
                        if usecase_py.startswith('usecase'):
                            usecase = usecase_py.replace('.py', '')
                            imported_module = import_module(
                                '.'.join([repository, process, usecase]))
                            imported_usecase = getattr(
                                imported_module, 'Study')()
                            imported_usecase.set_dump_directory(
                                dump_dir)
                            imported_usecase.setup_usecase()
                            if hasattr(imported_usecase, 'dspace'):
                                if type(imported_usecase.dspace) == dict:
                                    for key in imported_usecase.dspace.keys():
                                        if key != 'dspace_size':
                                            for key_2 in key_to_check_list:
                                                check = type(
                                                    imported_usecase.dspace[key][key_2]) == list
                                                self.assertTrue(
                                                    check, f'the usecase {repository}.{process}.{usecase} contains non list in its dspace : see {key}{key_2}')
                                else:
                                    for key_2 in key_to_check_list:
                                        for element in imported_usecase.dspace[key_2]:
                                            check = type(element) == list
                                            self.assertTrue(
                                                check, f'the usecase {repository}.{process}.{usecase} non list in its dspace : see {key_2}')
                            else:
                                print(
                                    f'no dpsace to examine in {repository}.{process}.{usecase}')
                else:
                    print(
                        f"Process {'.'.join([repository, process])} skipped. Check presence of __init__.py in the folder.")
