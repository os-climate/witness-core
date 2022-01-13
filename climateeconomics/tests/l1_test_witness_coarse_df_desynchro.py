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
from os.path import join, dirname, exists
import numpy as np
import pandas as pd

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest
from climateeconomics.sos_processes.iam.witness.witness_coarse_optim_process.usecase_witness_optim_invest_distrib import Study as witness_proc_usecase
import unittest
from energy_models.core.energy_study_manager import DEFAULT_COARSE_TECHNO_DICT


class WitnessCoarseJacobianDiscTest(AbstractJacobianUnittest):

    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    obj_const = ['welfare_objective', 'min_utility_objective', 'temperature_objective', 'CO2_objective', 'ppm_objective',
                 'total_prod_minus_min_prod_constraint_df', 'co2_emissions_objective', 'energy_production_objective', 'syngas_prod_objective', 'land_demand_constraint_df']

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        '''

        '''
        return [
        ]

    def test_01_desynchro(self):
        '''
        Test on the witness full MDA + design var to get bspline with func manager 

        we can test only lagrangian objective vs design var
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_coarse_optim_process', techno_dict=DEFAULT_COARSE_TECHNO_DICT, one_invest_discipline=True)
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_proc_usecase(
            bspline=True, execution_engine=self.ee, techno_dict=DEFAULT_COARSE_TECHNO_DICT, one_invest_discipline=True)
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
        full_values_dict['Test.WITNESS_MDO.WITNESS_Eval.max_mda_iter'] = 3
        full_values_dict['Test.WITNESS_MDO.max_iter'] = 2
        self.ee.load_study_from_input_dict(full_values_dict)

        disc = self.ee.root_process.sos_disciplines[0]

        self.ee.display_treeview_nodes()
        disc_techno = self.ee.root_process.sos_disciplines[0]

        self.ee.execute()
        df_coupled = self.ee.dm.get_value(
            'Test.WITNESS_MDO.WITNESS_Eval.WITNESS.temperature_df')
        df_ncoupled = self.ee.dm.get_value(
            'Test.WITNESS_MDO.WITNESS_Eval.WITNESS.Temperature_change.temperature_detail_df')
        self.assertListEqual(list(df_ncoupled['temp_atmo'].values), list(
            df_coupled['temp_atmo'].values), msg="desynchro of dataframes detected")


if '__main__' == __name__:
    cls = WitnessCoarseJacobianDiscTest()
    cls.test_01_desynchro()
