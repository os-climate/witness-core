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

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    Study as witness_sub_proc_usecase,
)


class WitnessFullGradient(AbstractJacobianUnittest):

    obj_const = [GlossaryCore.WelfareObjective, 'temperature_objective', 'CO2_objective', 'ppm_objective',
                 'co2_emissions_objective',
                 'CO2_tax_minus_CO2_damage_constraint_df', 'primary_energies_production',
                 'CO2_tax_minus_CCS_constraint_df', 'land_demand_constraint_df']

    def setUp(self):
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_01_gradient_objective_wrt_design_var_on_witness_full,
            # self.test_02_gradient_residus_wrt_state_var_on_witness_full,
            # self.test_03_gradient_residus_wrt_design_var_on_witness_full,
            # self.test_04_gradient_objective_wrt_design_var_on_witness_full
        ]

    def test_01_gradient_objective_wrt_design_var_on_witness_full(self):
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        repo = 'climateeconomics.sos_processes.iam.witness'
        chain_builders = self.ee.factory.get_builder_from_process(
            repo, 'witness_optim_sub_process')
        ns_dict = {GlossaryCore.NS_FUNCTIONS: f'{self.ee.study_name}',
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
        # values_dict['Test.WITNESS_Eval.max_mda_iter'] = 1
        values_dict['Test.WITNESS_Eval.WITNESS.EnergyMix.methane.FossilGas.methane_FossilGas_array_mix'] = 81 * [30.]
        self.ee.load_study_from_input_dict(values_dict)

        self.ee.load_study_from_input_dict(values_dict)

        output_full_names = ['Test.WITNESS_Eval.WITNESS.invest_objective_sum']
        # ['Test.WITNESS_Eval.WITNESS.EnergyMix.methane.FossilGas.methane_FossilGas_array_mix']
        input_full_names = [
            'Test.WITNESS_Eval.WITNESS.EnergyMix.methane.FossilGas.methane_FossilGas_array_mix']

        disc = self.ee.root_process.proxy_disciplines[0]
        disc.add_differentiated_inputs(input_full_names)
        disc.add_differentiated_outputs(output_full_names)

        dict_lin = disc.linearize()

        print(dict_lin)


if '__main__' == __name__:
    cls = WitnessFullGradient()
    cls.test_01_gradient_objective_wrt_design_var_on_witness_full()
