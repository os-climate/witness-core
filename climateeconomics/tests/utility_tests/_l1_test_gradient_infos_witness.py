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
from os.path import join, dirname
import numpy as np
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest
from climateeconomics.sos_processes.iam.witness.witness.usecase_witness import Study as witness_usecase
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import Study as witness_sub_proc_usecase


class WitnessFullJacobianDiscTest(AbstractJacobianUnittest):

    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    obj_const = ['welfare_objective', 'temperature_objective', 'CO2_objective', 'ppm_objective', 'co2_emissions_objective', 'CO2_tax_minus_CO2_damage_constraint_df', 'EnergyMix.methane.demand_violation', 'EnergyMix.hydrogen.gaseous_hydrogen.demand_violation', 'EnergyMix.biogas.demand_violation', 'EnergyMix.syngas.demand_violation', 'EnergyMix.liquid_fuel.demand_violation',
                 'EnergyMix.solid_fuel.demand_violation', 'EnergyMix.biomass_dry.demand_violation', 'EnergyMix.electricity.demand_violation', 'EnergyMix.biodiesel.demand_violation', 'EnergyMix.hydrogen.liquid_hydrogen.demand_violation', 'primary_energies_production', 'CO2_tax_minus_CCS_constraint_df', 'land_demand_constraint_df']

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_05_adjoint_with_bsplines,
            # self.test_02_gradient_residus_wrt_state_var_on_witness_full,
            # self.test_03_gradient_residus_wrt_design_var_on_witness_full,
            # self.test_04_gradient_objective_wrt_design_var_on_witness_full
        ]

    def test_01_gradient_objective_wrt_state_var_on_witness_full(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        repo = 'climateeconomics.sos_processes.iam.witness'
        chain_builders = self.ee.factory.get_builder_from_process(
            repo, 'witness_optim_sub_process')

        ns_dict = {'ns_functions': f'{self.ee.study_name}',
                   'ns_optim': f'{self.ee.study_name}',
                   'ns_public': f'{self.ee.study_name}', }
        self.ee.ns_manager.add_ns_def(ns_dict)

        self.ee.factory.set_builders_to_coupling_builder(
            chain_builders)

        self.ee.configure()

        usecase = witness_sub_proc_usecase(execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = {}
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)


        values_dict['Test.WITNESS_Eval.sub_mda_class'] = 'MDAGaussSeidel'
        values_dict['Test.WITNESS_Eval.max_mda_iter'] = 1

        self.ee.load_study_from_input_dict(values_dict)

        output_full_names = ['Test.WITNESS_Eval.WITNESS.invest_objective']
        input_full_names = ['Test.WITNESS_Eval.WITNESS.EnergyMix.methane.FossilGas.methane_FossilGas_array_mix']

        disc = self.ee.root_process
        disc.add_differentiated_inputs(input_full_names)
        disc.add_differentiated_outputs(output_full_names)

        dict_lin = disc.linearize()

        print(dict_lin)


if '__main__' == __name__:
    cls = WitnessFullJacobianDiscTest()
    cls.test_05_adjoint_with_bsplines()
