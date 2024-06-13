'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/19-2023/11/03 Copyright 2023 Capgemini

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
from copy import deepcopy
from tempfile import gettempdir

import numpy as np

from climateeconomics.sos_processes.iam.witness.witness_coarse.usecase_witness_coarse_new import (
    Study,
)
from gemseo.utils.compare_data_manager_tooling import (
    compare_dict,
    delete_keys_from_dict,
)
from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class WITNESSParallelTest(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.root_dir = gettempdir()
        self.ee = ExecutionEngine(self.name)

    def test_01_exec_parallel(self):
        """
        8 proc
        """
        n_proc = 16
        repo = 'climateeconomics.sos_processes.iam.witness'
        self.ee8 = ExecutionEngine(self.name)
        builder = self.ee8.factory.get_builder_from_process(
            repo, 'witness_coarse')

        self.ee8.factory.set_builders_to_coupling_builder(builder)
        self.ee8.configure()
        self.ee8.display_treeview_nodes()
        usecase = Study()
        usecase.study_name = self.name
        values_dict = {}
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)

        values_dict[f'{self.name}.sub_mda_class'] = "GSPureNewtonMDA"
        values_dict[f'{self.name}.max_mda_iter'] = 50
        values_dict[f'{self.name}.n_processes'] = n_proc

        self.ee8.load_study_from_input_dict(values_dict)

        self.ee8.execute()

        dm_dict_8 = deepcopy(self.ee8.get_anonimated_data_dict())

        """
        1 proc
        """
        n_proc = 1

        builder = self.ee.factory.get_builder_from_process(
            repo, 'witness_coarse')

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()
        self.ee.display_treeview_nodes()
        usecase = Study()
        usecase.study_name = self.name
        values_dict = {}

        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)
        values_dict[f'{self.name}.sub_mda_class'] = "GSPureNewtonMDA"
        values_dict[f'{self.name}.max_mda_iter'] = 50
        values_dict[f'{self.name}.n_processes'] = n_proc

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()
        dm_dict_1 = deepcopy(self.ee.get_anonimated_data_dict())
        residual_history = self.ee.root_process.sub_mda_list[0].residual_history

        dict_error = {}
        # to delete modelorigin and discipline dependencies which are not the
        # same
        delete_keys_from_dict(dm_dict_1)
        delete_keys_from_dict(dm_dict_8)
        compare_dict(dm_dict_1,
                     dm_dict_8, '', dict_error)

        residual_history8 = self.ee8.root_process.sub_mda_list[0].residual_history
        # self.assertListEqual(residual_history, residual_history8)
        for key, value in dict_error.items():
            print(key)
            print(value)

        for disc1, disc2 in zip(self.ee.root_process.proxy_disciplines, self.ee8.root_process.proxy_disciplines):
            if disc1.jac is not None:
                # print(disc1)
                for keyout, subjac in disc1.jac.items():
                    for keyin in subjac.keys():
                        comparison = disc1.jac[keyout][keyin].toarray(
                        ) == disc2.jac[keyout][keyin].toarray()
                        try:
                            self.assertTrue(comparison.all())
                        except:
                            print('error in jac')
                            print(keyout + ' vs ' + keyin)
                            np.set_printoptions(threshold=1e6)
                            for arr, arr2 in zip(disc1.jac[keyout][keyin], disc2.jac[keyout][keyin]):
                                if not (arr.toarray() == arr2.toarray()).all():
                                    print(arr)
                                    print(arr2)
        # The only different value is n_processes
        self.assertDictEqual(dict_error, {
            '.<study_ph>.n_processes.value': "1 and 16 don't match"})


if '__main__' == __name__:
    cls = WITNESSParallelTest()
    cls.setUp()
    cls.test_01_exec_parallel()
