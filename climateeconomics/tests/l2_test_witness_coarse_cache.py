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

import unittest

from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.core.energy_study_manager import DEFAULT_COARSE_TECHNO_DICT
from gemseo.utils.compare_data_manager_tooling import compare_dict
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.sos_processes.iam.witness.witness_coarse_optim_process.usecase_witness_optim_invest_distrib import (
    Study as witness_proc_usecase,
)


class WitnessCoarseCache(unittest.TestCase):

    def test_01_cache_on_witness_coarse_optim_with_unconverged_mda(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_coarse_optim_process', techno_dict=DEFAULT_COARSE_TECHNO_DICT,
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[1])
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_proc_usecase(
            bspline=True, execution_engine=self.ee, techno_dict=DEFAULT_COARSE_TECHNO_DICT, invest_discipline=INVEST_DISCIPLINE_OPTIONS[1])
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        algo_options = {"ftol_rel": 3e-16,
                        "normalize_design_space": False,
                        "maxls": 2 * 55,
                        "maxcor": 55,
                        "pg_tol": 1.e-8,
                        "max_iter": 2,
                        "disp": 110}
        full_values_dict['Test.WITNESS_MDO.algo_options'] = algo_options
        full_values_dict['Test.WITNESS_MDO.WITNESS_Eval.warm_start'] = False
        full_values_dict['Test.WITNESS_MDO.max_iter'] = 1
        full_values_dict['Test.WITNESS_MDO.WITNESS_Eval.sub_mda_class'] = 'MDAGaussSeidel'
        full_values_dict['Test.WITNESS_MDO.WITNESS_Eval.max_mda_iter'] = 1

        # execute optim without cache and retrieve dm
        self.ee.load_study_from_input_dict(full_values_dict)
        self.ee.execute()

        dm_without_cache = self.ee.dm.get_data_dict_values()

        # execute optim with SimpleCache and retrieve dm

        self.ee2 = ExecutionEngine(self.name)

        builder = self.ee2.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_coarse_optim_process', techno_dict=DEFAULT_COARSE_TECHNO_DICT,
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[1])
        self.ee2.factory.set_builders_to_coupling_builder(builder)
        self.ee2.configure()

        for cache_type_key in self.ee2.dm.get_all_namespaces_from_var_name('cache_type'):
            full_values_dict[cache_type_key] = 'SimpleCache'
        self.ee2.load_study_from_input_dict(full_values_dict)
        self.ee2.execute()

        dm_with_simple_cache = self.ee2.dm.get_data_dict_values()

        # remove cache_type keys from dm_with_simple_cache and dm_without_cache
        for cache_type_key in self.ee.dm.get_all_namespaces_from_var_name('cache_type') + self.ee.dm.get_all_namespaces_from_var_name('residuals_history'):
            dm_with_simple_cache.pop(cache_type_key)
            dm_without_cache.pop(cache_type_key)
        optim_output_df_simple_cache = dm_with_simple_cache.pop(
            'Test.WITNESS_MDO.WITNESS_Eval.FunctionsManager.optim_output_df')
        optim_output_df_simple_cache = optim_output_df_simple_cache.iloc[-1].drop(
            'iteration')
        optim_output_df_without_cache = dm_without_cache.pop(
            'Test.WITNESS_MDO.WITNESS_Eval.FunctionsManager.optim_output_df')
        optim_output_df_without_cache = optim_output_df_without_cache.iloc[-1].drop(
            'iteration')

        # compare values in dm_with_simple_cache and dm_without_cache
        dict_error = {}
        compare_dict(dm_with_simple_cache,
                     dm_without_cache, '', dict_error)
        self.assertDictEqual(dict_error, {})
        self.assertTrue(optim_output_df_simple_cache.equals(
            optim_output_df_without_cache))

    def test_02_cache_on_witness_coarse_optim(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_coarse_optim_process', techno_dict=DEFAULT_COARSE_TECHNO_DICT,
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[1])
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_proc_usecase(
            bspline=True, execution_engine=self.ee, techno_dict=DEFAULT_COARSE_TECHNO_DICT, invest_discipline=INVEST_DISCIPLINE_OPTIONS[1])
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        algo_options = {"ftol_rel": 3e-16,
                        "normalize_design_space": False,
                        "maxls": 2 * 55,
                        "maxcor": 55,
                        "pg_tol": 1.e-8,
                        "max_iter": 2,
                        "disp": 110}
        full_values_dict['Test.WITNESS_MDO.algo_options'] = algo_options
        full_values_dict['Test.WITNESS_MDO.WITNESS_Eval.warm_start'] = False
        full_values_dict['Test.WITNESS_MDO.max_iter'] = 2
        full_values_dict['Test.WITNESS_MDO.WITNESS_Eval.sub_mda_class'] = 'MDAGaussSeidel'
        full_values_dict['Test.WITNESS_MDO.WITNESS_Eval.max_mda_iter'] = 10

        # execute optim without cache and retrieve dm
        self.ee.load_study_from_input_dict(full_values_dict)
        self.ee.execute()

        dm_without_cache = self.ee.dm.get_data_dict_values()

        # execute optim with SimpleCache and retrieve dm

        self.ee2 = ExecutionEngine(self.name)

        builder = self.ee2.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_coarse_optim_process', techno_dict=DEFAULT_COARSE_TECHNO_DICT,
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[1])
        self.ee2.factory.set_builders_to_coupling_builder(builder)
        self.ee2.configure()

        for cache_type_key in self.ee2.dm.get_all_namespaces_from_var_name('cache_type'):
            full_values_dict[cache_type_key] = 'SimpleCache'
        self.ee2.load_study_from_input_dict(full_values_dict)
        self.ee2.execute()

        dm_with_simple_cache = self.ee2.dm.get_data_dict_values()

        # remove cache_type keys from dm_with_simple_cache and dm_without_cache
        for cache_type_key in self.ee.dm.get_all_namespaces_from_var_name('cache_type') + self.ee.dm.get_all_namespaces_from_var_name('residuals_history'):
            dm_with_simple_cache.pop(cache_type_key)
            dm_without_cache.pop(cache_type_key)
        optim_output_df_simple_cache = dm_with_simple_cache.pop(
            'Test.WITNESS_MDO.WITNESS_Eval.FunctionsManager.optim_output_df')
        optim_output_df_simple_cache = optim_output_df_simple_cache.iloc[-1].drop(
            'iteration')
        optim_output_df_without_cache = dm_without_cache.pop(
            'Test.WITNESS_MDO.WITNESS_Eval.FunctionsManager.optim_output_df')
        optim_output_df_without_cache = optim_output_df_without_cache.iloc[-1].drop(
            'iteration')

        # compare values in dm_with_simple_cache and dm_without_cache
        dict_error = {}
        compare_dict(dm_with_simple_cache,
                     dm_without_cache, '', dict_error)
        self.assertDictEqual(dict_error, {})
        self.assertTrue(optim_output_df_simple_cache.equals(
            optim_output_df_without_cache))


if '__main__' == __name__:
    cls = WitnessCoarseCache()
#     cls.test_01_cache_on_witness_coarse_optim_with_unconverged_mda()
    cls.test_02_cache_on_witness_coarse_optim()

