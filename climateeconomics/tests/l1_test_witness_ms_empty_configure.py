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
import unittest
from climateeconomics.sos_processes.iam.witness.witness_coarse.usecase_witness_coarse_new import Study
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from tempfile import gettempdir
from copy import deepcopy
from gemseo.utils.compare_data_manager_tooling import delete_keys_from_dict,\
    compare_dict
import numpy as np


class WITNESSEmptyMSStudyLoadingTest(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.root_dir = gettempdir()
        self.ee = ExecutionEngine(self.name)
        self.scatter_scenario = 'optimization scenarios'
        
    def load_empty_ms(self, proc_name):
        """
        loads a multiscenario with a list of empty scenarios
        """
        repo = 'climateeconomics.sos_processes.iam.witness'
        self.ee = ExecutionEngine(self.name)
        builder = self.ee.factory.get_builder_from_process(repo, proc_name)

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()
        
        self.ee.display_treeview_nodes()
        
        values_dict = {}
        values_dict[f'{self.name}.{self.scatter_scenario}.scenario_list'] = ["sc1", "sc2"]

        self.ee.load_study_from_input_dict(values_dict)
        
    def test_empty_ms_study_loading(self):
        """
        loops on the load empty ms function to test loading of multiscenario with empty scenarios
        """
        
        proc_list = ["witness_coarse_ms_optim_process",
                     "witness_coarse_dev_ms_optim_process",
                     "witness_ms_optim_process"]
        
        for proc in proc_list:
            print("\nLoad MS process <%s>"%proc)
            self.load_empty_ms(proc)


if '__main__' == __name__:

    cls = WITNESSEmptyMSStudyLoadingTest()
    cls.setUp()
    cls.test_empty_ms_study_loading()
