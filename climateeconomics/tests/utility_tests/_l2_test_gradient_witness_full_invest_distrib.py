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
from os.path import dirname, exists, join

import numpy as np
import pandas as pd
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    Study as witness_sub_proc_usecase,
)


class WitnessFullJacobianDiscTest(AbstractJacobianUnittest):
    obj_const = [GlossaryCore.WelfareObjective, 'min_utility_objective', 'temperature_objective', 'CO2_objective',
                 'ppm_objective', 'co2_emissions_objective', 'CO2_tax_minus_CO2_damage_constraint_df',
                 'EnergyMix.methane.demand_violation', 'EnergyMix.hydrogen.gaseous_hydrogen.demand_violation',
                 'EnergyMix.biogas.demand_violation', 'EnergyMix.syngas.demand_violation',
                 'EnergyMix.liquid_fuel.demand_violation',
                 'EnergyMix.solid_fuel.demand_violation', 'EnergyMix.biomass_dry.demand_violation',
                 'EnergyMix.electricity.demand_violation', 'EnergyMix.biodiesel.demand_violation',
                 'EnergyMix.hydrogen.liquid_hydrogen.demand_violation', 'primary_energies_production',
                 'CO2_tax_minus_CCS_constraint_df', 'land_demand_constraint_df']

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        '''

        '''
        return [
            #             self.test_02_gradient_objective_constraint_wrt_design_var_on_witness_full_subprocess_wofuncmanager,
            #             self.test_03_gradient_lagrangian_objective_wrt_design_var_on_witness_full_subprocess,
            self.test_05_gradient_lagrangian_objective_wrt_csv_design_var_on_witness_full_subprocess_each_step
        ]

    def test_05_gradient_lagrangian_objective_wrt_csv_design_var_on_witness_full_subprocess_each_step(self):
        '''
        Test on the witness full MDA + design var to get bspline with func manager 

        we can test only lagrangian objective vs design var

        Need to checkout to gems_without_cache in gems repository 
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        techno_dict = GlossaryEnergy.DEFAULT_TECHNO_DICT

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_optim_sub_process', techno_dict=techno_dict,
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[1])
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[1], execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        full_values_dict[f'{self.name}.{usecase.coupling_name}.tolerance_linear_solver_MDO'] = 1.0e-12
        full_values_dict[f'{self.name}.{usecase.coupling_name}.linearization_mode'] = 'adjoint'
        full_values_dict[f'{self.name}.{usecase.coupling_name}.warm_start'] = False
        full_values_dict[f'{self.name}.{usecase.coupling_name}.tolerance'] = 1.0e-12
        full_values_dict[f'{self.name}.{usecase.coupling_name}.chain_linearize'] = False
        full_values_dict[f'{self.name}.{usecase.coupling_name}.inner_mda_name'] = 'GSNewtonMDA'
        full_values_dict[f'{self.name}.{usecase.coupling_name}.max_mda_iter'] = 1
        self.ee.load_study_from_input_dict(full_values_dict)

        disc = self.ee.root_process.proxy_disciplines[0]

        values_dict_design_var = {}
        df_xvect = pd.read_csv(
            join(dirname(__file__), 'data', 'design_space_last_ite.csv'))
        for i, row in df_xvect.iterrows():
            try:
                ns_var = self.ee.dm.get_all_namespaces_from_var_name(
                    row['variable'])[0]
                values_dict_design_var[ns_var] = np.asarray(
                    row['value'][1:-1].split(', '), dtype=float)
            except:
                pass
        dspace_df = df_xvect

        self.ee.load_study_from_input_dict(values_dict_design_var)

        self.ee.execute()

        i = 49

        for disc in self.ee.root_process.proxy_disciplines[0].proxy_disciplines:
            #         disc = self.ee.dm.get_disciplines_with_name(
            #             f'{self.name}.{usecase.coupling_name}.WITNESS.EnergyMix')[0]
            outputs = disc.get_output_data_names()
            outputs = [output for output in outputs if self.ee.dm.get_data(
                output, 'coupling')]

            if disc.name == 'FunctionsManager':
                outputs.append(self.ee.dm.get_all_namespaces_from_var_name(
                    'objective_lagrangian')[0])
            inputs = disc.get_input_data_names()
            inputs = [input for input in inputs if self.ee.dm.get_data(input, 'coupling')
                      and not input.endswith(GlossaryCore.ResourcesPriceValue)
                      and not input.endswith('resources_CO2_emissions')]
            print(disc.name)
            print(i)
            if i > 73:

                print(inputs)
                print(outputs)
                pkl_name = f'jacobian_lagrangian_objective_wrt_design_var_on_witness_full_invest_distrib_{i}.pkl'
                filepath = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY,
                                pkl_name)
                if len(inputs) != 0:

                    if not exists(filepath):
                        self.ee.dm.delete_complex_in_df_and_arrays()
                        self.override_dump_jacobian = True
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-8, local_data={},
                                            inputs=inputs,
                                            outputs=outputs)  # filepath=filepath)
                    else:
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-8, local_data={},
                                            inputs=inputs,
                                            outputs=outputs)  # filepath=filepath)
            i += 1


if '__main__' == __name__:
    cls = WitnessFullJacobianDiscTest()
    cls.test_05_gradient_lagrangian_objective_wrt_csv_design_var_on_witness_full_subprocess_each_step()
    # self.test_06_gradient_each_discipline_on_dm_pkl()
