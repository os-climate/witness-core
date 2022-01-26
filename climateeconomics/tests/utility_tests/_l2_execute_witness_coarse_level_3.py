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
from climateeconomics.sos_processes.iam.witness.witness_coarse_process_one_distrib_3.usecase_witness_optim_invest_distrib import Study
from energy_models.core.energy_study_manager import DEFAULT_COARSE_TECHNO_DICT_ccs_3
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS


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

    def test_01_execute_coarse_level3(self):
        '''
        Test on the witness full MDA + design var to get bspline with func manager 

        we can test only lagrangian objective vs design var

        Need to checkout to gems_without_cache in gems repository 
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_coarse_process_one_distrib_3', techno_dict=DEFAULT_COARSE_TECHNO_DICT_ccs_3,
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[1])
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()
        usecase = Study(execution_engine=self.ee, run_usecase=True)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        input_dict_to_load = {}

        for uc_d in values_dict:
            input_dict_to_load.update(uc_d)

        self.ee.load_study_from_input_dict(input_dict_to_load)
        self.ee.execute()


if '__main__' == __name__:
    cls = WitnessCoarseJacobianDiscTest()
    cls.test_01_execute_coarse_level3()
