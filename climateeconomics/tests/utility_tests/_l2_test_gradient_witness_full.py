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
from climateeconomics.sos_processes.iam.witness.witness.usecase_witness import (
    Study as witness_usecase,
)
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
            #             self.test_05_gradient_lagrangian_objective_wrt_csv_design_var_on_witness_full_subprocess_each_step,
            self.test_06_gradient_lagrangian_objective_wrt_csv_design_var_on_crashed_x,
            self.test_07_gradient_all_disciplines_on_crashed_x,
        ]

    def _test_01_gradient_objective_wrt_state_var_on_witness_full_mda(self):
        '''
        ON WITNESS full MDA without design var and func manager
        test all constraint and objective vs variables out from design var
        Problem : NO design var then no bspline takes a long time
        '''
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

        values_dict[f'{self.name}.epsilon0'] = 1.0
        values_dict[f'{self.name}.tolerance_linear_solver_MDO'] = 1.0e-12
        values_dict[f'{self.name}.linearization_mode'] = 'adjoint'
        values_dict[f'{self.name}.tolerance'] = 1.0e-10
        values_dict[f'{self.name}.sub_mda_class'] = 'MDAGaussSeidel'

        self.ee.load_study_from_input_dict(values_dict)

        output_full_names = [f'{self.name}.{obj}' for obj in self.obj_const]
        input_full_names = [f'{self.name}.EnergyMix.invest_energy_mix',
                            f'{self.name}.{GlossaryCore.CO2TaxesValue}',
                            f'{self.name}.livestock_usage_factor_df']
        for energy in usecase.energy_list:
            input_full_names.append(
                f'{self.name}.EnergyMix.{energy}.{GlossaryCore.InvestLevelValue}')

        disc = self.ee.root_process

        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_objective_wrt_state_var_on_witness_full.pkl', local_data={},
                            discipline=disc, inputs=input_full_names,
                            outputs=output_full_names, derr_approx='complex_step', step=1.0e-12, parallel=True)

    def test_02_gradient_objective_constraint_wrt_design_var_on_witness_full_subprocess_wofuncmanager(self):
        '''
        Test on the witness full MDA + design var to get bspline without func manager
        If strong coupling we cannot check the adjoint then if we delete the func manager
        we can test over all constraint and objectives with the efficiency of bsplines compared to test 1
        '''
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

        usecase = witness_sub_proc_usecase(
            bspline=True, execution_engine=self.ee)
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
        full_values_dict[f'{self.name}.{usecase.coupling_name}.sub_mda_class'] = 'MDAGaussSeidel'
        self.ee.load_study_from_input_dict(full_values_dict)

        disc = self.ee.root_process.proxy_disciplines[0]

        output_full_names = [
            f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.{obj}' for obj in self.obj_const]
        input_full_names = [
            f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CO2_taxes_array',
            f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.livestock_usage_factor_array']
        for energy in full_values_dict[f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.{GlossaryCore.energy_list}']:
            energy_wo_dot = energy.replace('.', '_')
            input_full_names.append(
                f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.{energy_wo_dot}_array_mix')

            for technology in full_values_dict[
                f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.{GlossaryCore.techno_list}']:
                technology_wo_dot = technology.replace('.', '_')
                input_full_names.append(
                    f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix')

        for energy in full_values_dict[f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.{GlossaryCore.ccs_list}']:
            energy_wo_dot = energy.replace('.', '_')
            input_full_names.append(
                f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CCUS.{energy}.{energy_wo_dot}_array_mix')

            for technology in full_values_dict[
                f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CCUS.{energy}.{GlossaryCore.techno_list}']:
                technology_wo_dot = technology.replace('.', '_')
                input_full_names.append(
                    f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CCUS.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix')

        istart = 10
        iend = istart + 10
        if iend >= len(input_full_names):
            iend = len(input_full_names)
        self.check_jacobian(location=dirname(__file__),
                            filename=f'jacobian_objectives_constraint_wrt_design_var_{istart}_{iend}_on_witness_full.pkl',
                            discipline=disc,
                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                            inputs=input_full_names[istart:iend],
                            outputs=output_full_names, parallel=True)

    def test_03_gradient_lagrangian_objective_wrt_design_var_on_witness_full_subprocess(self):
        '''
        Test on the witness full MDA + design var to get bspline with func manager

        we can test only lagrangian objective vs design var
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_optim_sub_process')
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(
            bspline=True, execution_engine=self.ee)
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
        full_values_dict[f'{self.name}.{usecase.coupling_name}.sub_mda_class'] = 'MDAGaussSeidel'
        self.ee.load_study_from_input_dict(full_values_dict)

        disc = self.ee.root_process.proxy_disciplines[0]

        output_full_names = [
            f'{self.name}.objective_lagrangian']
        input_full_names = [
            f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CO2_taxes_array',
            f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.livestock_usage_factor_array']
        for energy in full_values_dict[f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.{GlossaryCore.energy_list}']:
            energy_wo_dot = energy.replace('.', '_')
            input_full_names.append(
                f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.{energy_wo_dot}_array_mix')

            for technology in full_values_dict[
                f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.{GlossaryCore.techno_list}']:
                technology_wo_dot = technology.replace('.', '_')
                input_full_names.append(
                    f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix')

        for energy in full_values_dict[f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.{GlossaryCore.ccs_list}']:
            energy_wo_dot = energy.replace('.', '_')
            input_full_names.append(
                f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CCUS.{energy}.{energy_wo_dot}_array_mix')

            for technology in full_values_dict[
                f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CCUS.{energy}.{GlossaryCore.techno_list}']:
                technology_wo_dot = technology.replace('.', '_')
                input_full_names.append(
                    f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CCUS.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix')

        self.ee.display_treeview_nodes(display_variables=True)
        istart = 10
        iend = istart + 10
        if iend >= len(input_full_names):
            iend = len(input_full_names)
        self.check_jacobian(location=dirname(__file__),
                            filename=f'jacobian_lagrangian_objective_wrt_design_var_{istart}_{iend}_on_witness_full.pkl',
                            discipline=disc,
                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                            inputs=input_full_names[istart:iend],
                            outputs=output_full_names, parallel=True)
        if disc.jac is not None:
            print(disc.jac[output_full_names[0]])

    def test_04_gradient_lagrangian_objective_wrt_csv_design_var_on_witness_full_subprocess(self):
        '''
        Test on the witness full MDA + design var to get bspline with func manager

        we can test only lagrangian objective vs design var
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_optim_sub_process')
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(
            bspline=True, execution_engine=self.ee)
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
        full_values_dict[f'{self.name}.{usecase.coupling_name}.sub_mda_class'] = 'MDAGaussSeidel'

        self.ee.load_study_from_input_dict(full_values_dict)

        disc = self.ee.root_process.proxy_disciplines[0]

        values_dict_design_var = {}
        df_xvect = pd.read_csv(
            join(dirname(__file__), 'data', 'design_space_last_ite_crash.csv'))
        for i, row in df_xvect.iterrows():
            ns_var = self.ee.dm.get_all_namespaces_from_var_name(
                row['variable'])[0]
            values_dict_design_var[ns_var] = np.asarray(
                row['value'][1:-1].split(', '), dtype=float)
        dspace_df = df_xvect

        self.ee.load_study_from_input_dict(values_dict_design_var)

        output_full_names = [
            f'{self.name}.objective_lagrangian']
        input_full_names = [
            f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CO2_taxes_array',
            f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.livestock_usage_factor_array']

        for energy in full_values_dict[f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.{GlossaryCore.energy_list}']:
            energy_wo_dot = energy.replace('.', '_')
            input_full_names.append(
                f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.{energy_wo_dot}_array_mix')

            for technology in full_values_dict[
                f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.{GlossaryCore.techno_list}']:
                technology_wo_dot = technology.replace('.', '_')
                input_full_names.append(
                    f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix')

        for energy in full_values_dict[f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.{GlossaryCore.ccs_list}']:
            energy_wo_dot = energy.replace('.', '_')
            input_full_names.append(
                f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CCUS.{energy}.{energy_wo_dot}_array_mix')

            for technology in full_values_dict[
                f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CCUS.{energy}.{GlossaryCore.techno_list}']:
                technology_wo_dot = technology.replace('.', '_')
                input_full_names.append(
                    f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CCUS.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix')

        self.ee.display_treeview_nodes(display_variables=True)
        istart = 0
        iend = istart + 50
        if iend >= len(input_full_names):
            iend = len(input_full_names)
        self.check_jacobian(location=dirname(__file__),
                            filename=f'jacobian_lagrangian_objective_wrt_design_var_{istart}_{iend}_on_witness_full_withx0csv_crash.pkl',
                            discipline=disc,
                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                            inputs=input_full_names[istart:iend],
                            outputs=output_full_names, parallel=True)
        if disc.jac is not None:
            print(disc.jac[output_full_names[0]])

    def test_05_gradient_lagrangian_objective_wrt_csv_design_var_on_witness_full_subprocess_each_step(self):
        '''
        Test on the witness full MDA + design var to get bspline with func manager

        we can test only lagrangian objective vs design var

        Need to checkout to gems_without_cache in gems repository
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_optim_sub_process')
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(
            bspline=True, execution_engine=self.ee)
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
        full_values_dict[f'{self.name}.{usecase.coupling_name}.sub_mda_class'] = 'GSNewtonMDA'
        full_values_dict[f'{self.name}.{usecase.coupling_name}.max_mda_iter'] = 2
        full_values_dict[f'{self.name}.{usecase.coupling_name}.WITNESS.CCUS.carbon_capture.flue_gas_effect'] = False

        self.ee.load_study_from_input_dict(full_values_dict)

        disc = self.ee.root_process.proxy_disciplines[0]

        values_dict_design_var = {}
        df_xvect = pd.read_csv(
            join(dirname(__file__), 'data', 'design_space_last_ite_full_wo_effect.csv'))
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

        #         output_full_names = [
        #             f'{self.name}.objective_lagrangian']
        #         input_full_names = [
        #             f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CO2_taxes_array',
        #             f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.livestock_usage_factor_array']
        #
        #         for energy in full_values_dict[f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.{GlossaryCore.energy_list}']:
        #             energy_wo_dot = energy.replace('.', '_')
        #             input_full_names.append(
        #                 f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.{energy_wo_dot}_array_mix')
        #
        #             for technology in full_values_dict[f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.{GlossaryCore.techno_list}']:
        #                 technology_wo_dot = technology.replace('.', '_')
        #                 input_full_names.append(
        #                     f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix')
        #
        #         for energy in full_values_dict[f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.{GlossaryCore.ccs_list}']:
        #             energy_wo_dot = energy.replace('.', '_')
        #             input_full_names.append(
        #                 f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CCUS.{energy}.{energy_wo_dot}_array_mix')
        #
        #             for technology in full_values_dict[f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CCUS.{energy}.{GlossaryCore.techno_list}']:
        #                 technology_wo_dot = technology.replace('.', '_')
        #                 input_full_names.append(
        #                     f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CCUS.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix')

        #         values_dict_step = {}
        #         values_dict_step[f'{self.name}.{usecase.coupling_name}.sub_mda_class'] = 'GSNewtonMDA'
        #         values_dict_step[f'{self.name}.{usecase.coupling_name}.max_mda_iter_gs'] = 2
        #         values_dict_step[f'{self.name}.{usecase.coupling_name}.max_mda_iter'] = step + 2
        #         self.ee.load_study_from_input_dict(values_dict_step)
        self.ee.execute()
        ns = self.ee.dm.get_all_namespaces_from_var_name(GlossaryCore.energy_list)[0]
        energy_list = self.ee.dm.get_value(ns)

        inputs_names = [
            f'{self.name}.{usecase.coupling_name}.WITNESS.EnergyMix.{energy}.energy_prices' for energy in energy_list if
            energy not in [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]]
        inputs_names.extend([
            f'{self.name}.{usecase.coupling_name}.WITNESS.EnergyMix.{energy}.{GlossaryCore.EnergyProductionValue}' for energy in
            energy_list if energy not in [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])
        inputs_names.extend(
            [f'{self.name}.{usecase.coupling_name}.WITNESS.EnergyMix.{energy}.energy_consumption' for energy in
             energy_list if energy not in [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])
        inputs_names.extend(
            [f'{self.name}.{usecase.coupling_name}.WITNESS.CCUS.{energy}.energy_consumption' for energy in
             [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])
        inputs_names.extend(
            [f'{self.name}.{usecase.coupling_name}.WITNESS.CCUS.{energy}.{GlossaryCore.EnergyProductionValue}' for energy in
             [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])
        inputs_names.extend([
            f'{self.name}.{usecase.coupling_name}.WITNESS.CCUS.{energy}.energy_prices' for energy in
            [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])
        inputs_names.extend(
            [f'{self.name}.{usecase.coupling_name}.WITNESS.EnergyMix.syngas.syngas_ratio'])
        i = 0

        for disc in self.ee.root_process.proxy_disciplines[0].proxy_disciplines:
            #         disc = self.ee.dm.get_disciplines_with_name(
            #             f'{self.name}.{usecase.coupling_name}.WITNESS.EnergyMix')[0]
            outputs = disc.get_output_data_names()
            outputs = [output for output in outputs if self.ee.dm.get_data(output, 'coupling')
                       and not output.endswith('all_streams_demand_ratio')]

            if disc.name == 'FunctionsManager':
                outputs.append(self.ee.dm.get_all_namespaces_from_var_name(
                    'objective_lagrangian')[0])
            inputs = disc.get_input_data_names()
            inputs = [input for input in inputs if self.ee.dm.get_data(input, 'coupling')
                      and not input.endswith(GlossaryCore.ResourcesPriceValue)
                      and not input.endswith('resources_CO2_emissions')
                      and not input.endswith('all_streams_demand_ratio')]
            print(disc.name)
            print(i)
            if i not in [63, 64]:

                print(inputs)
                print(outputs)
                pkl_name = f'jacobian_lagrangian_objective_wrt_design_var_on_witness_full_wo_fg_effect_{i}.pkl'
                filepath = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY, 'l2_witness_full',
                                pkl_name)
                if len(inputs) != 0:

                    if not exists(filepath):
                        self.ee.dm.delete_complex_in_df_and_arrays()
                        self.override_dump_jacobian = True
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                                            inputs=inputs,
                                            outputs=outputs)  # , filepath=filepath)
                    else:
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                                            inputs=inputs,
                                            outputs=outputs)  # , filepath=filepath)
            i += 1

    def test_06_gradient_each_discipline_on_dm_pkl(self):
        '''
        Test on the witness full MDA + design var to get bspline with func manager

        we can test only lagrangian objective vs design var
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_optim_sub_process')
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        pkl_dict = pd.read_pickle(
            join(dirname(__file__), 'data', 'dm_3_ite.pkl'))
        inp_dict = {key.replace('<study_ph>.WITNESS_MDO',
                                self.name): value for key, value in pkl_dict.items()}

        self.ee.load_study_from_dict(inp_dict)
        i = 0

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
            inputs = [input for input in inputs if self.ee.dm.get_data(
                input, 'coupling') and not input.endswith(GlossaryCore.ResourcesPriceValue) and not input.endswith(
                'resources_CO2_emissions')]
            print(disc.name)
            print(i)
            if i not in [6, 27, 53, 58, 62]:

                print(inputs)
                print(outputs)
                pkl_name = f'jacobian_lagrangian_objective_wrt_design_var_on_witness_full_withx0_ite3nR_{i}.pkl'
                filepath = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY, 'l2_witness_full',
                                pkl_name)
                if len(inputs) != 0:

                    if not exists(filepath):

                        self.ee.dm.delete_complex_in_df_and_arrays()

                        self.override_dump_jacobian = True
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                                            inputs=inputs,
                                            outputs=outputs)  # , filepath=filepath)
                    else:
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                                            inputs=inputs,
                                            outputs=outputs)  # , filepath=filepath)
            i += 1

    def test_06_gradient_lagrangian_objective_wrt_csv_design_var_on_crashed_x(self):
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
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[1], execution_engine=self.ee, techno_dict=techno_dict,
            bspline=True)
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
        full_values_dict[f'{self.name}.{usecase.coupling_name}.sub_mda_class'] = 'MDAGaussSeidel'
        full_values_dict[f'{self.name}.{usecase.coupling_name}.max_mda_iter'] = 200
        self.ee.load_study_from_input_dict(full_values_dict)
        # self.ee.execute()
        disc = self.ee.root_process.proxy_disciplines[0]

        values_dict_design_var = {}
        df_xvect = pd.read_csv(
            join(dirname(__file__), 'data', 'design_space_last_ite_coarse_fail.csv'))
        for i, row in df_xvect.iterrows():
            try:
                ns_var = self.ee.dm.get_all_namespaces_from_var_name(
                    row['variable'])[0]
                values_dict_design_var[ns_var] = np.asarray(
                    row['value'][1:-1].split(', '), dtype=float)
            except:
                pass

        self.ee.load_study_from_input_dict(values_dict_design_var)
        self.ee.set_debug_mode('min_max_grad')
        self.ee.execute()
        # --------------------#
        # ---    ADJOINT   ---#
        # --------------------#

        output_full_names = [
            f'{self.name}.objective_lagrangian']
        input_full_names = [
            f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.livestock_usage_factor_array']

        for energy in full_values_dict[f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.{GlossaryCore.energy_list}']:
            energy_wo_dot = energy.replace('.', '_')
            for technology in full_values_dict[
                f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.{GlossaryCore.techno_list}']:
                technology_wo_dot = technology.replace('.', '_')
                input_full_names.append(
                    f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.EnergyMix.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix')

        for energy in full_values_dict[f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.{GlossaryCore.ccs_list}']:
            energy_wo_dot = energy.replace('.', '_')
            for technology in full_values_dict[
                f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CCUS.{energy}.{GlossaryCore.techno_list}']:
                technology_wo_dot = technology.replace('.', '_')
                input_full_names.append(
                    f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.CCUS.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix')

        pkl_name = 'jacobian_lagrangian_objective_wrt_design_var_on_witness_full_invest_distrib_x.pkl'
        self.ee.display_treeview_nodes(display_variables=True)
        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                            step=1.0e-15, derr_approx='complex_step', threshold=1e-8, local_data={},
                            inputs=input_full_names, outputs=output_full_names)

    def test_07_gradient_all_disciplines_on_crashed_x(self):
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        techno_dict = GlossaryEnergy.DEFAULT_TECHNO_DICT

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_optim_sub_process', techno_dict=techno_dict,
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[1])
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[1], execution_engine=self.ee, techno_dict=techno_dict,
            bspline=True)
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
        full_values_dict[f'{self.name}.{usecase.coupling_name}.sub_mda_class'] = 'MDAGaussSeidel'
        full_values_dict[f'{self.name}.{usecase.coupling_name}.max_mda_iter'] = 200
        self.ee.load_study_from_input_dict(full_values_dict)
        # self.ee.execute()
        disc = self.ee.root_process.proxy_disciplines[0]

        values_dict_design_var = {}
        df_xvect = pd.read_csv(
            join(dirname(__file__), 'data', 'design_space_last_ite_coarse_fail.csv'))
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
        self.ee.set_debug_mode('min_max_grad')
        self.ee.execute()

        # -------------------------#
        # ---    disc by disc   ---#
        # -------------------------#

        for disc in self.ee.root_process.proxy_disciplines[0].proxy_disciplines:
            if disc.name != 'WITNESS.EnergyMix.liquid_fuel.Refinery':
                continue
            #         disc = self.ee.dm.get_disciplines_with_name(
            #             f'{self.name}.{usecase.coupling_name}.WITNESS.EnergyMix')[0]
            outputs = disc.get_output_data_names()
            outputs = [output for output in outputs if self.ee.dm.get_data(
                output, 'coupling')]

            if disc.name == 'FunctionsManager':
                outputs.append(self.ee.dm.get_all_namespaces_from_var_name(
                    'objective_lagrangian')[0])
            inputs = disc.get_input_data_names()
            inputs = [input for input in inputs if self.ee.dm.get_data(
                input, 'coupling') and not input.endswith(GlossaryCore.ResourcesPriceValue) and not input.endswith(
                'resources_CO2_emissions')]
            print(disc.name)
            print(i)
            pkl_name = 'pickle_discilpine.pkl'
            filepath = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY, 'l2_witness_full',
                            pkl_name)

            print('--------------------------------------------------------------------')
            print('First check the whole df')
            print('--------------------------------------------------------------------')
            np.set_printoptions(threshold=100000000)
            if len(inputs) != 0:
                self.ee.dm.delete_complex_in_df_and_arrays()
                self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                    step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                                    inputs=[
                                        f'Test.WITNESS_Eval.WITNESS.EnergyMix.liquid_fuel.{GlossaryEnergy.Refinery}.{GlossaryCore.InvestLevelValue}'],
                                    outputs=[
                                        'Test.WITNESS_Eval.WITNESS.EnergyMix.liquid_fuel.{GlossaryEnergy.Refinery}.techno_production'])

            print('--------------------------------------------------------------------')
            print('Then check col by col')
            print('--------------------------------------------------------------------')
            for output_column in disc.get_sosdisc_outputs('techno_production').columns[1:]:
                for input_column in disc.get_sosdisc_inputs(GlossaryCore.InvestLevelValue).columns[1:]:
                    self.ee.dm.delete_complex_in_df_and_arrays()
                    print('input_column : ', input_column)
                    print('output_columns : ', output_column)
                    self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                        step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                                        inputs=[
                                            f'Test.WITNESS_Eval.WITNESS.EnergyMix.liquid_fuel.{GlossaryEnergy.Refinery}.{GlossaryCore.InvestLevelValue}'],
                                        input_column=input_column,
                                        outputs=[
                                            'Test.WITNESS_Eval.WITNESS.EnergyMix.liquid_fuel.{GlossaryEnergy.Refinery}.techno_production'],
                                        output_column=output_column)


if '__main__' == __name__:
    cls = WitnessFullJacobianDiscTest()
    cls.test_06_gradient_lagrangian_objective_wrt_csv_design_var_on_crashed_x()
    cls.test_07_gradient_all_disciplines_on_crashed_x()
