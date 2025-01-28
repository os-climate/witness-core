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
from os.path import dirname

import numpy as np
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness.usecase_witness import (
    Study as witness_usecase,
)
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    Study as witness_sub_proc_usecase,
)


class WitnessFullJacobianDiscTest(AbstractJacobianUnittest):

    obj_const = [GlossaryCore.NegativeWelfareObjective, 'temperature_objective', 'CO2_objective', 'ppm_objective',
                 'co2_emissions_objective',
                 'CO2_tax_minus_CO2_damage_constraint_df', 'primary_energies_production',
                 'CO2_tax_minus_CCS_constraint_df', 'land_demand_constraint_df']

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
            repo, 'witness')

        ns_dict = {GlossaryCore.NS_FUNCTIONS: f'{self.ee.study_name}',
                   'ns_optim': f'{self.ee.study_name}',
                   'ns_public': f'{self.ee.study_name}', }
        self.ee.ns_manager.add_ns_def(ns_dict)

        self.ee.factory.set_builders_to_coupling_builder(
            chain_builders)

        self.ee.configure()

        usecase = witness_usecase(execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = {}
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)

        values_dict['Test.epsilon0'] = 1.0
        values_dict['Test.tolerance_linear_solver_MDO'] = 1.0e-12
        values_dict['Test.linearization_mode'] = 'adjoint'
        values_dict['Test.tolerance'] = 1.0e-10
        values_dict['Test.inner_mda_name'] = 'MDAGaussSeidel'

        self.ee.load_study_from_input_dict(values_dict)

        output_full_names = [f'Test.{obj}' for obj in self.obj_const]
        input_full_names = ['Test.EnergyMix.invest_energy_mix',
                            f'Test.{GlossaryCore.CO2TaxesValue}']

        self.ee.display_treeview_nodes()
        disc = self.ee.root_process

        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_objective_wrt_state_var_on_witness_full.pkl', discipline=disc,
                            inputs=input_full_names,
                            outputs=output_full_names, derr_approx='complex_step', local_data={}, step=1.0e-12,
                            parallel=True)

    def test_02_gradient_residus_wrt_state_var_on_witness_full(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        repo = 'climateeconomics.sos_processes.iam.witness'
        chain_builders = self.ee.factory.get_builder_from_process(
            repo, 'witness')

        ns_dict = {GlossaryCore.NS_FUNCTIONS: f'{self.ee.study_name}',
                   'ns_optim': f'{self.ee.study_name}',
                   'ns_public': f'{self.ee.study_name}', }
        self.ee.ns_manager.add_ns_def(ns_dict)

        self.ee.factory.set_builders_to_coupling_builder(
            chain_builders)

        self.ee.configure()

        usecase = witness_usecase(execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = {}
        for dict_item in usecase.setup_usecase():
            values_dict.update(dict_item)

        values_dict['Test.epsilon0'] = 1.0
        values_dict['Test.tolerance_linear_solver_MDO'] = 1.0e-8
        values_dict['Test.linearization_mode'] = 'adjoint'
        values_dict['Test.tolerance'] = 1.0e-10
        values_dict['Test.inner_mda_name'] = 'MDAGaussSeidel'

        self.ee.load_study_from_input_dict(values_dict)

        disc = self.ee.root_process

        output_full_names = [f'Test.{GlossaryCore.TemperatureDfValue}', f'Test.{GlossaryCore.UtilityDfValue}', f'Test.{GlossaryCore.EconomicsDfValue}',
                             f'Test.{GlossaryCore.CarbonCycleDfValue}', 'Test.CO2_emissions_df', f'Test.{GlossaryCore.DamageFractionDfValue}',
                             f'Test.EnergyMix.{GlossaryCore.StreamProductionValue}', f'Test.EnergyMix.{GlossaryCore.EnergyInvestmentsValue}',
                             f'Test.EnergyMix.{GlossaryCore.CO2EmissionsGtValue}', f'Test.EnergyMix.{GlossaryCore.EnergyMeanPriceValue}']

        input_full_names = ['Test.EnergyMix.invest_energy_mix',
                            f'Test.{GlossaryCore.CO2TaxesValue}']
        input_full_names.extend(
            [f'Test.EnergyMix.{energy}.invest_techno_mix' for energy in usecase.energy_list])

        self.check_jacobian(location=dirname(__file__), filename='jacobian_residus_wrt_state_var_on_witness_full.pkl',
                            discipline=disc, inputs=input_full_names, local_data={},
                            outputs=output_full_names, derr_approx='complex_step', step=1.0e-15, parallel=True)

    def test_03_gradient_residus_wrt_design_var_on_witness_full(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        coupling_name = "WITNESS_Eval"
        designvariable_name = "DesignVariables"
        func_manager_name = "FunctionsManager"
        extra_name = 'WITNESS'
        # retrieve energy process
        chain_builders = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness')

        # modify namespaces defined in the child process
        self.ee.ns_manager.update_namespace_list_with_extra_ns(
            extra_name, after_name=self.ee.study_name)
        self.ee.factory.update_builder_list_with_extra_name(
            extra_name, builder_list=chain_builders)

        # design variables builder
        design_var_path = 'climateeconomics.core.design_variables_translation.witness.design_var_disc.Design_Var_Discipline'
        design_var_builder = self.ee.factory.get_builder_from_module(
            f'{designvariable_name}', design_var_path)
        chain_builders.append(design_var_builder)

        #         # function manager builder
        #         fmanager_path = 'sos_trades_core.execution_engine.func_manager.func_manager_disc.FunctionManagerDisc'
        #         fmanager_builder = self.ee.factory.get_builder_from_module(
        #             f'{func_manager_name}', fmanager_path)
        #         chain_builders.append(fmanager_builder)

        # modify namespaces defined in the child process
        self.ee.ns_manager.update_namespace_list_with_extra_ns(
            coupling_name, after_name=self.ee.study_name)

        ns_dict = {GlossaryCore.NS_FUNCTIONS: f'{self.ee.study_name}.{coupling_name}.{extra_name}',
                   'ns_public': f'{self.ee.study_name}',
                   'ns_optim': f'{self.ee.study_name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        # create coupling builder
        coupling_builder = self.ee.factory.create_builder_coupling(
            coupling_name)
        coupling_builder.set_builder_info('cls_builder', chain_builders)
        coupling_builder.set_builder_info('with_data_io', True)

        self.ee.factory.set_builders_to_coupling_builder(coupling_builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)

        full_values_dict[f'{usecase.study_name}.WITNESS_Eval.linear_solver_MDO_options'] = {'tol': 1.0e-14,
                                                                                            'max_iter': 50000}
        full_values_dict[f'{usecase.study_name}.WITNESS_Eval.linear_solver_MDA_options'] = {'tol': 1.0e-14,
                                                                                            'max_iter': 50000}
        full_values_dict[f'{usecase.study_name}.WITNESS_Eval.linearization_mode'] = 'adjoint'
        full_values_dict[f'{usecase.study_name}.WITNESS_Eval.tolerance'] = 1.0e-12
        full_values_dict[f'{usecase.study_name}.WITNESS_Eval.max_mda_iter'] = 200
        full_values_dict[f'{usecase.study_name}.WITNESS_Eval.inner_mda_name'] = 'MDAGaussSeidel'

        self.ee.load_study_from_input_dict(full_values_dict)

        disc = self.ee.root_process.proxy_disciplines[0]
        namespace = 'Test.WITNESS_Eval.WITNESS'
        output_full_names = [f'{namespace}.{GlossaryCore.TemperatureDfValue}', f'{namespace}.{GlossaryCore.UtilityDfValue}', f'{namespace}.{GlossaryCore.EconomicsDfValue}',
                             f'{namespace}.{GlossaryCore.CarbonCycleDfValue}', f'{namespace}.{GlossaryCore.CO2EmissionsDfValue}', f'{namespace}.{GlossaryCore.DamageFractionDfValue}',
                             f'{namespace}.EnergyMix.{GlossaryCore.StreamProductionValue}', f'{namespace}.EnergyMix.{GlossaryCore.EnergyInvestmentsValue}',
                             f'{namespace}.EnergyMix.{GlossaryCore.CO2EmissionsGtValue}', f'{namespace}.EnergyMix.{GlossaryCore.EnergyMeanPriceValue}',
                             f'{namespace}.CO2_objective', f'{namespace}.ppm_objective',
                             f'{namespace}.temperature_objective',
                             f'{namespace}.CO2_tax_minus_CO2_damage_constraint_df',
                             f'{namespace}.CO2_tax_minus_CCS_constraint_df']

        self.ee.display_treeview_nodes(display_variables=True)
        #         input_full_names = ['Test.WITNESS_Eval.CO2_taxes_array']
        #         for energy in full_values_dict[f'{self.name}.WITNESS_Eval.{GlossaryCore.energy_list}']:
        #             energy_wo_dot = energy.replace('.', '_')
        #             input_full_names.append(
        #                 f'{self.name}.WITNESS_Eval.DesignVariables.{energy}.{energy_wo_dot}_array_mix')

        # for technology in full_values_dict[f'Test.WITNESS_Eval.EnergyMix.{energy}.{GlossaryCore.techno_list}']:
        #     technology_wo_dot = technology.replace('.', '_')
        #     input_full_names.append(
        #         f'{self.name}.WITNESS_Eval.DesignVariables.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix')

        self.check_jacobian(location=dirname(__file__), filename='jacobian_residus_wrt_design_var_on_witness_full.pkl',
                            local_data={}, discipline=disc,
                            inputs=[f'{namespace}.EnergyMix.electricity.CoalGen.electricity_CoalGen_array_mix',
                                    f'{namespace}.EnergyMix.liquid_fuel.{GlossaryEnergy.Refinery}.liquid_fuel_Refinery_array_mix',
                                    f'{namespace}.CO2_taxes_array'],
                            outputs=output_full_names, derr_approx='complex_step', step=1.0e-15, parallel=True)

    def test_04_gradient_objective_wrt_design_var_on_witness_full(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_optim_sub_process')
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        full_values_dict['Test.epsilon0'] = 1.0
        # 1.0e-12
        full_values_dict['Test.WITNESS_Eval.tolerance_linear_solver_MDO'] = 1.0e-8
        full_values_dict['Test.WITNESS_Eval.linearization_mode'] = 'adjoint'
        full_values_dict['Test.WITNESS_Eval.tolerance'] = 1.0e-10
        full_values_dict['Test.WITNESS_Eval.warm_start'] = False
        full_values_dict['Test.WITNESS_Eval.inner_mda_name'] = 'MDAGaussSeidel'
        self.ee.load_study_from_input_dict(full_values_dict)

        disc = self.ee.root_process.proxy_disciplines[0]

        output_full_names = [
            f'Test.WITNESS_Eval.{obj}' for obj in self.obj_const]
        input_full_names = ['Test.WITNESS_Eval.CO2_taxes_array']
        for energy in full_values_dict[f'{self.name}.WITNESS_Eval.{GlossaryCore.energy_list}']:
            energy_wo_dot = energy.replace('.', '_')
            input_full_names.append(
                f'{self.name}.WITNESS_Eval.DesignVariables.{energy}.{energy_wo_dot}_array_mix')

        #             for technology in full_values_dict[f'Test.WITNESS_Eval.EnergyMix.{energy}.{GlossaryCore.techno_list}']:
        #                 technology_wo_dot = technology.replace('.', '_')
        #                 input_full_names.append(
        #                     f'{self.name}.WITNESS_Eval.DesignVariables.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix')

        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_objective_wrt_design_var_on_witness_full.pkl', discipline=disc,
                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                            inputs=input_full_names,
                            outputs=output_full_names, parallel=True)

    def test_05_adjoint_with_bsplines(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        coupling_name = "WITNESS_Eval"
        designvariable_name = "DesignVariables"
        extra_name = 'WITNESS'
        # retrieve energy process
        chain_builders = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness')

        # modify namespaces defined in the child process
        self.ee.ns_manager.update_namespace_list_with_extra_ns(
            extra_name, after_name=self.ee.study_name)
        self.ee.factory.update_builder_list_with_extra_name(
            extra_name, builder_list=chain_builders)

        # design variables builder
        design_var_path = 'climateeconomics.core.design_variables_translation.witness_bspline.design_var_disc.Design_Var_Discipline'
        design_var_builder = self.ee.factory.get_builder_from_module(
            f'{designvariable_name}', design_var_path)
        chain_builders.append(design_var_builder)

        #         # function manager builder
        #         fmanager_path = 'sos_trades_core.execution_engine.func_manager.func_manager_disc.FunctionManagerDisc'
        #         fmanager_builder = self.ee.factory.get_builder_from_module(
        #             f'{func_manager_name}', fmanager_path)
        #         chain_builders.append(fmanager_builder)

        # modify namespaces defined in the child process
        self.ee.ns_manager.update_namespace_list_with_extra_ns(
            coupling_name, after_name=self.ee.study_name)

        ns_dict = {GlossaryCore.NS_FUNCTIONS: f'{self.ee.study_name}.{coupling_name}.{extra_name}',
                   'ns_public': f'{self.ee.study_name}',
                   'ns_optim': f'{self.ee.study_name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        # create coupling builder
        coupling_builder = self.ee.factory.create_builder_coupling(
            coupling_name)
        coupling_builder.set_builder_info('cls_builder', chain_builders)
        coupling_builder.set_builder_info('with_data_io', True)

        self.ee.factory.set_builders_to_coupling_builder(coupling_builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)

        full_values_dict[f'{usecase.study_name}.WITNESS_Eval.linear_solver_MDO_options'] = {'tol': 1.0e-14,
                                                                                            'max_iter': 50000}
        full_values_dict[f'{usecase.study_name}.WITNESS_Eval.linear_solver_MDA_options'] = {'tol': 1.0e-14,
                                                                                            'max_iter': 50000}
        full_values_dict[f'{usecase.study_name}.WITNESS_Eval.linearization_mode'] = 'adjoint'
        full_values_dict[f'{usecase.study_name}.WITNESS_Eval.tolerance'] = 1.0e-12
        full_values_dict[f'{usecase.study_name}.WITNESS_Eval.max_mda_iter'] = 200
        full_values_dict[f'{usecase.study_name}.WITNESS_Eval.inner_mda_name'] = 'MDAGaussSeidel'

        input_full_names = ['Test.WITNESS_Eval.WITNESS.CO2_taxes_array']
        nb_poles = 5
        for energy in full_values_dict[f'{self.name}.WITNESS_Eval.WITNESS.{GlossaryCore.energy_list}']:
            energy_wo_dot = energy.replace('.', '_')
            input_name = f'{self.name}.WITNESS_Eval.WITNESS.EnergyMix.{energy}.{energy_wo_dot}_array_mix'
            input_full_names.append(input_name)
            full_values_dict[input_name] = np.linspace(1, 2, nb_poles)

            for technology in full_values_dict[
                f'{self.name}.WITNESS_Eval.WITNESS.EnergyMix.{energy}.{GlossaryCore.techno_list}']:
                technology_wo_dot = technology.replace('.', '_')
                input_name = f'{self.name}.WITNESS_Eval.WITNESS.EnergyMix.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix'
                input_full_names.append(input_name)
                full_values_dict[input_name] = np.linspace(3, 4, nb_poles)
        self.ee.load_study_from_input_dict(full_values_dict)

        disc = self.ee.root_process.proxy_disciplines[0]
        namespace = 'Test.WITNESS_Eval.WITNESS'
        output_full_names = [f'{namespace}.{GlossaryCore.TemperatureDfValue}', f'{namespace}.{GlossaryCore.UtilityDfValue}', f'{namespace}.{GlossaryCore.EconomicsDfValue}',
                             f'{namespace}.{GlossaryCore.CarbonCycleDfValue}', f'{namespace}.{GlossaryCore.CO2EmissionsDfValue}', f'{namespace}.{GlossaryCore.DamageFractionDfValue}',
                             f'{namespace}.EnergyMix.{GlossaryCore.StreamProductionValue}', f'{namespace}.EnergyMix{GlossaryCore.EnergyInvestmentsValue}',
                             f'{namespace}.EnergyMix.{GlossaryCore.CO2EmissionsGtValue}', f'{namespace}.EnergyMix.{GlossaryCore.EnergyMeanPriceValue}',
                             f'{namespace}.CO2_objective', f'{namespace}.ppm_objective',
                             f'{namespace}.utility_objective',
                             f'{namespace}.temperature_objective',
                             f'{namespace}.CO2_tax_minus_CO2_damage_constraint_df',
                             f'{namespace}.CO2_tax_minus_CCS_constraint_df']

        self.ee.display_treeview_nodes(display_variables=True)

        self.check_jacobian(location=dirname(__file__), filename='jacobian_adjoint_with_bsplines_witness_full.pkl',
                            discipline=disc, local_data={}, inputs=input_full_names,
                            outputs=output_full_names, derr_approx='complex_step', step=1.0e-15, parallel=True)


if '__main__' == __name__:
    cls = WitnessFullJacobianDiscTest()
    cls.test_05_adjoint_with_bsplines()
