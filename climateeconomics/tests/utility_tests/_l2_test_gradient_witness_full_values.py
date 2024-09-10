'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/26-2023/11/03 Copyright 2023 Capgemini

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
from os.path import dirname, exists, join

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore


class WitnessFullJacobianDiscTest(AbstractJacobianUnittest):

    obj_const = [GlossaryCore.WelfareObjective, 'min_utility_objective', 'temperature_objective', 'CO2_objective', 'ppm_objective', 'co2_emissions_objective', 'CO2_tax_minus_CO2_damage_constraint_df', 'EnergyMix.methane.demand_violation', 'EnergyMix.hydrogen.gaseous_hydrogen.demand_violation', 'EnergyMix.biogas.demand_violation', 'EnergyMix.syngas.demand_violation', 'EnergyMix.liquid_fuel.demand_violation',
                 'EnergyMix.solid_fuel.demand_violation', 'EnergyMix.biomass_dry.demand_violation', 'EnergyMix.electricity.demand_violation', 'EnergyMix.biodiesel.demand_violation', 'EnergyMix.hydrogen.liquid_hydrogen.demand_violation', 'primary_energies_production', 'CO2_tax_minus_CCS_constraint_df', 'land_demand_constraint_df']

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        '''

        '''
        return [
        ]

    def test_01(self):
        '''
        Test on the witness full MDA + design var to get bspline with func manager 

        we can test only lagrangian objective vs design var

        Need to checkout to gems_without_cache in gems repository 
        '''
        max_discipline = {}
        for i in range(79):
            pkl_name = f'jacobian_lagrangian_objective_wrt_design_var_on_witness_full_withx0csv_crash_{i}.pkl'
            filepath = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY, 'l2_witness_full',
                            pkl_name)
            if exists(filepath):
                print(pkl_name)
                pkl_dict = pd.read_pickle(filepath)
                try:
                    max_dict_discipline = {f'{key} vs {key2}': np.max(np.abs(
                        value2)) for key, value in pkl_dict.items() for key2, value2 in value.items()}
                    print(max_dict_discipline)
                    max_discipline[max(
                        max_dict_discipline, key=max_dict_discipline.get)] = max(
                        list(max_dict_discipline.values()))
                    print('max of discipline :', max(
                        list(max_dict_discipline.values())), max(
                        max_dict_discipline, key=max_dict_discipline.get))
                except:
                    pass
        print('max max max', max(list(max_discipline.values())), max(
            max_discipline, key=max_discipline.get))


if '__main__' == __name__:
    cls = WitnessFullJacobianDiscTest()
    cls.test_01()
    # self.test_06_gradient_each_discipline_on_dm_pkl()
