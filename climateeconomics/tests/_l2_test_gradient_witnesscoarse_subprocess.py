'''
Copyright 2023 Capgemini

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

import ast
from os.path import dirname, join

import pandas as pd
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.glossaryenergy import GlossaryEnergy
from numpy import array
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    Study as witness_sub_proc_usecase,
)


class OptimSubprocessJacobianDiscTest(AbstractJacobianUnittest):

    def analytic_grad_entry(self):
        return [self.test_01_gradient_subprocess_objective_over_design_var(),
                ]

    def setUp(self):
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_01_gradient_subprocess_objective_over_design_var(self):
        """
        """
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process('climateeconomics.sos_processes.iam.witness',
                                                           'witness_optim_sub_process',
                                                           techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT,
                                                           invest_discipline=INVEST_DISCIPLINE_OPTIONS[2],
                                                           process_level='dev')
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(bspline=False,
                                           execution_engine=self.ee,
                                           techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT,
                                           process_level='dev',
                                           )
        usecase.study_name = self.name
        usecase.init_from_subusecase = True
        directory = join(AbstractJacobianUnittest.PICKLE_DIRECTORY, 'optim_check_gradient_dev')

        values_dict = usecase.setup_usecase()
        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)

        # Do not use a gradient method to validate gradient is better, Gauss Seidel works
        full_values_dict[f'{usecase.study_name}.{usecase.coupling_name}.tolerance'] = 1.0e-12
        full_values_dict[f'{usecase.study_name}.{usecase.coupling_name}.sub_mda_class'] = 'MDAGaussSeidel'
        full_values_dict[f'{usecase.study_name}.{usecase.coupling_name}.max_mda_iter'] = 30
        full_values_dict[f'{usecase.study_name}.{usecase.coupling_name}.warm_start'] = False
        full_values_dict[
            f'{usecase.study_name}.{usecase.coupling_name}.{usecase.extra_name}.assumptions_dict'] = {
            'compute_gdp': False,
            'compute_climate_impact_on_gdp': False,
            'activate_climate_effect_population': False,
            'activate_pandemic_effects': False,
            'invest_co2_tax_in_renewables': False
        }
        full_values_dict[
            f"{usecase.study_name}.{usecase.coupling_name}.{usecase.extra_name}.ccs_price_percentage"] = 0.0
        full_values_dict[
            f"{usecase.study_name}.{usecase.coupling_name}.{usecase.extra_name}.co2_damage_price_percentage"] = 0.0
        self.ee.load_study_from_input_dict(full_values_dict)

        # Add design space to the study by filling design variables :
        design_space = pd.read_csv(join(dirname(__file__), 'design_space_uc1_500ites.csv'))
        design_space_values_dict = {}
        for variable in design_space['variable'].values:
            # value in design space is considered as string we need to transform it into array
            str_val = design_space[design_space['variable'] == variable]['value'].values[0]
            design_space_values_dict[self.ee.dm.get_all_namespaces_from_var_name(variable)[0]] = array(
                ast.literal_eval(str_val))

        self.ee.load_study_from_input_dict(design_space_values_dict)
        self.ee.execute()

        # loop over all disciplines

        coupling_disc = self.ee.root_process.proxy_disciplines[0]

        outputs = [self.ee.dm.get_all_namespaces_from_var_name(GlossaryCore.NegativeWelfareObjective)[0],
                   self.ee.dm.get_all_namespaces_from_var_name('emax_enet_constraint')[0],
                   self.ee.dm.get_all_namespaces_from_var_name('delta_capital_constraint')[0]]
        inputs_name = [f'{energy}_{techno}_array_mix' for energy, techno_dict in GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT.items() for
                       techno in techno_dict['value']]
        inputs_name = [name.replace('.', '_') for name in inputs_name]
        inputs = []
        for name in inputs_name:
            inputs.extend(self.ee.dm.get_all_namespaces_from_var_name(name))
        inputs = [
            'Test.WITNESS_Eval.WITNESS.CCUS.carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix']
        pkl_name = 'jacobian_obj_vs_design_var_witness_coarse_subprocess.pkl'

        # self.override_dump_jacobian = True
        self.check_jacobian(location=dirname(__file__), filename=pkl_name,
                            discipline=coupling_disc.mdo_discipline_wrapp.mdo_discipline,
                            step=1.0e-4, derr_approx='finite_differences', threshold=1e-15,
                            local_data=coupling_disc.mdo_discipline_wrapp.mdo_discipline.local_data,
                            inputs=inputs,
                            outputs=outputs)

    def test_02_gradient_subprocess_objective_over_design_var_for_all_iterations(self):
        """
        Check gradient of objective wrt design variables for all stored iterations of a previously runned study
        """
        # Add design space to the study by filling design variables :
        design_space = pd.read_csv(join(dirname(__file__), 'data',  'all_iteration_dict.csv'))
        all_iterations_dspace_list = [eval(dspace) for dspace in design_space['value'].values]
        iter = 0

        for dspace_dict in all_iterations_dspace_list[-5:]:

            self.name = 'Test'
            self.ee = ExecutionEngine(self.name)

            builder = self.ee.factory.get_builder_from_process('climateeconomics.sos_processes.iam.witness',
                                                               'witness_optim_sub_process',
                                                               techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT,
                                                               invest_discipline=INVEST_DISCIPLINE_OPTIONS[2],
                                                               process_level='dev')
            self.ee.factory.set_builders_to_coupling_builder(builder)
            self.ee.configure()

            usecase = witness_sub_proc_usecase(bspline=False,
                                               execution_engine=self.ee,
                                               techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT,
                                               process_level='dev',
                                               )
            usecase.study_name = self.name
            usecase.init_from_subusecase = True
            directory = join(AbstractJacobianUnittest.PICKLE_DIRECTORY, 'optim_check_gradient_dev')

            values_dict = usecase.setup_usecase()
            full_values_dict = {}
            for dict_v in values_dict:
                full_values_dict.update(dict_v)

            # Do not use a gradient method to validate gradient is better, Gauss Seidel works
            full_values_dict[f'{usecase.study_name}.{usecase.coupling_name}.tolerance'] = 1.0e-15
            full_values_dict[f'{usecase.study_name}.{usecase.coupling_name}.sub_mda_class'] = 'MDAGaussSeidel'
            full_values_dict[f'{usecase.study_name}.{usecase.coupling_name}.max_mda_iter'] = 30
            full_values_dict[f'{usecase.study_name}.{usecase.coupling_name}.warm_start'] = False
            # same hypothesis as uc1
            full_values_dict[
                f'{usecase.study_name}.{usecase.coupling_name}.{usecase.extra_name}.assumptions_dict'] = {
                'compute_gdp': False,
                'compute_climate_impact_on_gdp': False,
                'activate_climate_effect_population': False,
                'activate_pandemic_effects': False,
                'invest_co2_tax_in_renewables': False
            }
            full_values_dict[
                f"{usecase.study_name}.{usecase.coupling_name}.{usecase.extra_name}.ccs_price_percentage"] = 0.0
            full_values_dict[
                f"{usecase.study_name}.{usecase.coupling_name}.{usecase.extra_name}.co2_damage_price_percentage"] = 0.0
            self.ee.load_study_from_input_dict(full_values_dict)
            test_results = []
            self.ee.logger.info(f'testing iteration {iter}')
            design_space_values_dict = {}
            for variable_name, variable_value in dspace_dict.items():
                design_space_values_dict[self.ee.dm.get_all_namespaces_from_var_name(variable_name)[0]] = array(variable_value)
            design_space_values_dict.update(full_values_dict)
            self.ee.load_study_from_input_dict(design_space_values_dict)

            self.ee.update_from_dm()
            self.ee.prepare_execution()

            # loop over all disciplines

            coupling_disc = self.ee.root_process.proxy_disciplines[0]

            outputs = [self.ee.dm.get_all_namespaces_from_var_name(GlossaryCore.NegativeWelfareObjective)[0],
                       self.ee.dm.get_all_namespaces_from_var_name('objective_lagrangian')[0],
                       self.ee.dm.get_all_namespaces_from_var_name('Energy invest minimization objective')[0],
                       self.ee.dm.get_all_namespaces_from_var_name('last_year_discounted_utility_objective')[0],
                       self.ee.dm.get_all_namespaces_from_var_name('minimum_ppm_constraint')[0],
                       self.ee.dm.get_all_namespaces_from_var_name('Lower bound usable capital constraint')[0],
                       ]
            inputs_name = [f'{energy}_{techno}_array_mix' for energy, techno_dict in GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT.items() for
                           techno in techno_dict['value']]
            inputs_name = [name.replace('.', '_') for name in inputs_name]
            inputs = []
            for name in inputs_name:
                inputs.extend(self.ee.dm.get_all_namespaces_from_var_name(name))
            #inputs = [
            #    'Test.WITNESS_Eval.WITNESS.CCUS.carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix']
            pkl_name = f'jacobian_obj_vs_design_var_witness_coarse_subprocess_iter_{iter}.pkl'

            # store all these variables for next test
            """
            var_in_to_store = [self.ee.dm.get_all_namespaces_from_var_name('Energy invest minimization objective')[0],
                               self.ee.dm.get_all_namespaces_from_var_name('last_year_discounted_utility_objective')[0],
                               self.ee.dm.get_all_namespaces_from_var_name('minimum_ppm_constraint')[0],
                               self.ee.dm.get_all_namespaces_from_var_name('Lower bound usable capital constraint')[0],
                               self.ee.dm.get_all_namespaces_from_var_name('function_df')[0],
                               self.ee.dm.get_all_namespaces_from_var_name(GlossaryCore.NegativeWelfareObjective)[0],
                               self.ee.dm.get_all_namespaces_from_var_name('gwp20_objective')[0],
                               self.ee.dm.get_all_namespaces_from_var_name('gwp100_objective')[0],
                               self.ee.dm.get_all_namespaces_from_var_name('energy_wasted_objective')[0],
                               self.ee.dm.get_all_namespaces_from_var_name('rockstrom_limit_constraint')[0],
                               self.ee.dm.get_all_namespaces_from_var_name('minimum_ppm_constraint')[0],
                               self.ee.dm.get_all_namespaces_from_var_name('calories_per_day_constraint')[0],
                               self.ee.dm.get_all_namespaces_from_var_name('carbon_storage_constraint')[0],
                               self.ee.dm.get_all_namespaces_from_var_name('total_prod_minus_min_prod_constraint_df')[0],
                               self.ee.dm.get_all_namespaces_from_var_name('energy_production_objective')[0],
                               self.ee.dm.get_all_namespaces_from_var_name('syngas_prod_objective')[0],
                               self.ee.dm.get_all_namespaces_from_var_name('land_demand_constraint')[0],
                               self.ee.dm.get_all_namespaces_from_var_name('function_df')[0],
                               ]
            self.dict_val_updt = {}
            
            for elem in var_in_to_store:
                self.dict_val_updt.update({elem: self.ee.dm.get_value(elem)})
            """
            #self.ee.execute()
            dict_values_cleaned = {k: v for k, v in design_space_values_dict.items() if self.ee.dm.check_data_in_dm(k)}

            try:
                self.override_dump_jacobian = True
                self.check_jacobian(location=dirname(__file__), filename=pkl_name,
                                    discipline=coupling_disc.mdo_discipline_wrapp.mdo_discipline,
                                    step=1.0e-18, derr_approx='complex_step', threshold=1e-16,
                                    local_data=dict_values_cleaned,#coupling_disc.mdo_discipline_wrapp.mdo_discipline.local_data,#design_space_values_dict,
                                    inputs=inputs,
                                    outputs=outputs)
                test_results.append((iter, True))
                self.ee.logger.info(f'iteration {iter} succeeded')
            except AssertionError:
                test_results.append((iter, False))
                self.ee.logger.info(f'iteration {iter} failed')
            iter += 1
            self.ee.logger.info(f'Result of each iteration {test_results}')

    def _test_03_func_manager_w_point(self):
        """
        Test only func manager with a special point from test 02.
        This test cannont be executed without some modifications on FuncManager. Default type is dataframe if variable not in dm
        (like in witness) but we have arrays
        """

        # The test was developped to check gradient of func manager at same point as failure in test_02
        self.test_02_gradient_subprocess_objective_over_design_var_for_all_iterations()
        self.name = 'Test'
        # -- init the case
        func_mng_name = 'FunctionsManager'
        prefix = self.name + '.' + func_mng_name + '.'

        ee = ExecutionEngine(self.name)
        ns_dict = {GlossaryCore.NS_FUNCTIONS: self.name + '.' + 'WITNESS_Eval.WITNESS',
                   'ns_optim': self.name + '.' + func_mng_name}
        ee.ns_manager.add_ns_def(ns_dict)

        mod_list = 'sostrades_optimization_plugin.models.func_manager.func_manager_disc.FunctionManagerDisc'
        fm_builder = ee.factory.get_builder_from_module(
            'WITNESS_Eval.FunctionsManager', mod_list)
        ee.factory.set_builders_to_coupling_builder(fm_builder)
        ee.configure()
        # Test.WITNESS_Eval.FunctionManager.function_df
        # Test.WITNESS_Eval.FunctionManager.function_df
        ee.load_study_from_input_dict(self.dict_val_updt)
        ee.execute()

        # all inputs to test
        inputs = [ee.dm.get_all_namespaces_from_var_name('minimum_ppm_constraint')[0],
                  ee.dm.get_all_namespaces_from_var_name('Energy invest minimization objective')[0],
                  ee.dm.get_all_namespaces_from_var_name('last_year_discounted_utility_objective')[0],
                  ee.dm.get_all_namespaces_from_var_name('minimum_ppm_constraint')[0],
                  ee.dm.get_all_namespaces_from_var_name('Lower bound usable capital constraint')[0],
                  ee.dm.get_all_namespaces_from_var_name(GlossaryCore.NegativeWelfareObjective)[0],
                  ee.dm.get_all_namespaces_from_var_name('gwp20_objective')[0],
                  ee.dm.get_all_namespaces_from_var_name('gwp100_objective')[0],
                  ee.dm.get_all_namespaces_from_var_name('energy_wasted_objective')[0],
                  ee.dm.get_all_namespaces_from_var_name('rockstrom_limit_constraint')[0],
                  ee.dm.get_all_namespaces_from_var_name('minimum_ppm_constraint')[0],
                  ee.dm.get_all_namespaces_from_var_name('calories_per_day_constraint')[0],
                  ee.dm.get_all_namespaces_from_var_name('carbon_storage_constraint')[0],
                  ee.dm.get_all_namespaces_from_var_name('total_prod_minus_min_prod_constraint_df')[0],
                  ee.dm.get_all_namespaces_from_var_name('energy_production_objective')[0],
                  ee.dm.get_all_namespaces_from_var_name('syngas_prod_objective')[0],
                  ee.dm.get_all_namespaces_from_var_name('land_demand_constraint')[0],
                  ]
        outputs = [ee.dm.get_all_namespaces_from_var_name('objective_lagrangian')[0]]
        disc = ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        disc.check_jacobian(
            input_data=disc.local_data,
            threshold=1e-15, inputs=inputs, step=1e-4,
            outputs=outputs, derr_approx='finite_differences')

