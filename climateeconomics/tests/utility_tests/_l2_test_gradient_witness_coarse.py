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
from climateeconomics.sos_processes.iam.witness.witness_coarse.usecase_witness_coarse_new import (
    Study as witness_coarse_usecase,
)
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    Study as witness_sub_proc_usecase,
)


class WitnessCoarseJacobianDiscTest(AbstractJacobianUnittest):
    obj_const = [GlossaryCore.WelfareObjective, 'min_utility_objective', 'temperature_objective', 'CO2_objective',
                 'ppm_objective',
                 'total_prod_minus_min_prod_constraint_df', 'co2_emissions_objective', 'energy_production_objective',
                 'syngas_prod_objective', 'land_demand_constraint_df']

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        '''

        '''
        return [
        ]

    def test_02_gradient_objective_constraint_wrt_design_var_on_witness_coarse_subprocess_wofuncmanager(self):
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
            'climateeconomics.sos_processes.iam.witness', 'witness', techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT,
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[0])

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
            bspline=True, execution_engine=self.ee, techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT,
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[0])
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
            f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.{GlossaryCore.CO2TaxesValue}',
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

        disc_techno = self.ee.root_process.proxy_disciplines[0]

        disc_techno.check_jacobian(derr_approx='complex_step', inputs=input_full_names, local_data={},
                                   outputs=output_full_names,
                                   load_jac_path=join(dirname(__file__), 'jacobian_pkls',
                                                      'jacobian_objectives_constraint_wrt_design_var_on_witness_coarse.pkl'))

    def test_03_gradient_lagrangian_objective_wrt_design_var_on_witness_coarse_subprocess(self):
        '''
        Test on the witness full MDA + design var to get bspline with func manager 

        we can test only lagrangian objective vs design var
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness', techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT,
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[0])
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(
            bspline=True, execution_engine=self.ee, techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT,
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[0])
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
            f'{self.name}.{usecase.coupling_name}.{usecase.extra_name}.{GlossaryCore.CO2TaxesValue}',
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
        disc_techno = self.ee.root_process.proxy_disciplines[0]

        disc_techno.check_jacobian(derr_approx='complex_step', inputs=input_full_names, local_data={},
                                   outputs=output_full_names,
                                   load_jac_path=join(dirname(__file__), 'jacobian_pkls',
                                                      'jacobian_lagrangian_objective_wrt_design_var_on_witness_coarse.pkl'))

        if disc.jac is not None:
            print(disc.jac[output_full_names[0]])

    def test_05_gradient_witness_coarse_eachdiscipline(self):
        '''
        Test on the witness full MDA + design var to get bspline with func manager 

        we can test only lagrangian objective vs design var

        Need to checkout to gems_without_cache in gems repository 
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness', techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT,
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[0])
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_coarse_usecase(
            bspline=True, execution_engine=self.ee, techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT,
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[0])
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        full_values_dict[f'{self.name}.tolerance_linear_solver_MDO'] = 1.0e-12
        full_values_dict[f'{self.name}.linearization_mode'] = 'adjoint'
        full_values_dict[f'{self.name}.warm_start'] = False
        full_values_dict[f'{self.name}.tolerance'] = 1.0e-10
        full_values_dict[f'{self.name}.chain_linearize'] = False
        full_values_dict[f'{self.name}.sub_mda_class'] = 'GSNewtonMDA'
        full_values_dict[f'{self.name}.max_mda_iter'] = 1
        self.ee.load_study_from_input_dict(full_values_dict)

        # self.ee.execute()
        full_values_dict = {}
        full_values_dict[f'{self.name}.CCUS.ccs_percentage'] = pd.DataFrame(
            {GlossaryCore.Years: np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1), 'ccs_percentage': 25})
        full_values_dict[f'{self.name}.sub_mda_class'] = 'GSNewtonMDA'
        full_values_dict[f'{self.name}.max_mda_iter'] = 1
        self.ee.load_study_from_input_dict(full_values_dict)
        disc = self.ee.root_process.proxy_disciplines[0]

        values_dict_design_var = {}
        df_xvect = pd.read_csv(
            join(dirname(__file__), 'data', 'design_space_last_ite_coarse.csv'))
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

        i = 0

        self.ee.execute()
        ns = self.ee.dm.get_all_namespaces_from_var_name(GlossaryCore.energy_list)[0]
        energy_list = self.ee.dm.get_value(ns)

        inputs_names = [
            f'{self.name}.WITNESS_Eval.WITNESS.EnergyMix.{energy}.energy_prices' for energy in energy_list if
            energy not in [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]]
        inputs_names.extend([
            f'{self.name}.WITNESS_Eval.WITNESS.EnergyMix.{energy}.{GlossaryCore.EnergyProductionValue}' for energy in energy_list if
            energy not in [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])
        inputs_names.extend(
            [f'{self.name}.WITNESS_Eval.WITNESS.EnergyMix.{energy}.energy_consumption' for energy in energy_list if
             energy not in [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])
        inputs_names.extend(
            [f'{self.name}.WITNESS_Eval.WITNESS.CCUS.{energy}.energy_consumption' for energy in
             [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])
        inputs_names.extend(
            [f'{self.name}.WITNESS_Eval.WITNESS.CCUS.{energy}.{GlossaryCore.EnergyProductionValue}' for energy in
             [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])
        inputs_names.extend([
            f'{self.name}.WITNESS_Eval.WITNESS.CCUS.{energy}.energy_prices' for energy in
            [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])
        inputs_names.extend(
            [f'{self.name}.WITNESS_Eval.WITNESS.EnergyMix.syngas.syngas_ratio'])
        i = 0

        for disc in self.ee.root_process.proxy_disciplines:
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
                print('*********************')
                print(inputs)
                print(outputs)
                pkl_name = f'jacobian_lagrangian_objective_wrt_design_var_on_witness_full_withx0csv_crash_{i}.pkl'
                filepath = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY, 'l2_witness_full',
                                pkl_name)
                if len(inputs) != 0:

                    if not exists(filepath):
                        self.ee.dm.delete_complex_in_df_and_arrays()
                        self.override_dump_jacobian = True
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                                            inputs=inputs,
                                            outputs=outputs)
                    else:
                        self.check_jacobian(location=dirname(__file__), filename=pkl_name, discipline=disc,
                                            step=1.0e-15, derr_approx='complex_step', threshold=1e-5, local_data={},
                                            inputs=inputs,
                                            outputs=outputs)
            i += 1

    def test_06_gradient_witness_coarse_subprocess_each_discipline(self):
        '''
        Test on the witness full MDA + design var to get bspline with func manager 

        we can test only lagrangian objective vs design var

        Need to checkout to gems_without_cache in gems repository 
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_optim_sub_process',
            techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT, one_invest_discipline=True)
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(
            execution_engine=self.ee, techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT,
            invest_discipline=INVEST_DISCIPLINE_OPTIONS[0])
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)
        full_values_dict[f'{self.name}.tolerance_linear_solver_MDO'] = 1.0e-12
        full_values_dict[f'{self.name}.linearization_mode'] = 'adjoint'
        full_values_dict[f'{self.name}.warm_start'] = False
        full_values_dict[f'{self.name}.tolerance'] = 1.0e-10
        full_values_dict[f'{self.name}.chain_linearize'] = False
        full_values_dict[f'{self.name}.WITNESS_Eval.sub_mda_class'] = 'GSNewtonMDA'
        full_values_dict[f'{self.name}.WITNESS_Eval.max_mda_iter'] = 2
        self.ee.load_study_from_input_dict(full_values_dict)

        values_dict_design_var = {}
        df_xvect = pd.read_csv(
            join(dirname(__file__), 'data', 'design_space_last_ite_coarse.csv'))
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

        i = 0

        for disc in self.ee.root_process.proxy_disciplines[0].proxy_disciplines:
            outputs = disc.get_output_data_names()
            outputs = [output for output in outputs if self.ee.dm.get_data(
                output, 'coupling')]

            inputs = disc.get_input_data_names()
            inputs = [input for input in inputs if self.ee.dm.get_data(
                input, 'coupling')]
            print(disc.name)
            print(i)
            if i not in []:

                print(inputs)
                print(outputs)
                pkl_name = f'jacobian_witness_coarse_subprocess_optim_eachdiscipline_{i}.pkl'
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

    def test_06_gradient_witness_coarse_subprocess_each_discipline_bis(self):
        '''
        Test on the witness full MDA + design var to get bspline with func manager 

        we can test only lagrangian objective vs design var

        Need to checkout to gems_without_cache in gems repository 
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_coarse_process_one_distrib',
            techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT, one_invest_discipline=True)
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        pkl_dict = pd.read_pickle(
            join(dirname(__file__), 'data', 'dm_crash.pkl'))
        inp_dict = {key.replace('usecase_witness_optim_invest_distrib',
                                self.name): value for key, value in pkl_dict.items()}

        self.ee.load_study_from_dict(inp_dict)

        i = 0

        for disc in self.ee.root_process.proxy_disciplines[0].proxy_disciplines:
            outputs = disc.get_output_data_names()
            outputs = [output for output in outputs if self.ee.dm.get_data(
                output, 'coupling')]

            inputs = disc.get_input_data_names()
            inputs = [input for input in inputs if self.ee.dm.get_data(
                input, 'coupling')]
            print(disc.name)
            print(i)
            if i not in []:

                print(inputs)
                print(outputs)
                pkl_name = 'jacobian_witness_coarse_subprocess_optim_eachdiscipline_adjoint.pkl'
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


if '__main__' == __name__:
    cls = WitnessCoarseJacobianDiscTest()
    cls.test_06_gradient_witness_coarse_subprocess_each_discipline_bis()
