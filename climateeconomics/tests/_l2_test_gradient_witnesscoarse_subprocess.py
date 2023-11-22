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

from os.path import join, dirname
import pandas as pd
from numpy import array
import ast

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest

from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import \
    Study as witness_sub_proc_usecase
from energy_models.core.energy_study_manager import DEFAULT_COARSE_TECHNO_DICT
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS


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
                                                           techno_dict=DEFAULT_COARSE_TECHNO_DICT,
                                                           invest_discipline=INVEST_DISCIPLINE_OPTIONS[2],
                                                           process_level='dev')
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(bspline=False,
                                           execution_engine=self.ee,
                                           techno_dict=DEFAULT_COARSE_TECHNO_DICT,
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
        inputs_name = [f'{energy}_{techno}_array_mix' for energy, techno_dict in DEFAULT_COARSE_TECHNO_DICT.items() for
                       techno in techno_dict['value']]
        inputs_name = [name.replace('.', '_') for name in inputs_name]
        inputs = []
        for name in inputs_name:
            inputs.extend(self.ee.dm.get_all_namespaces_from_var_name(name))
        inputs = [
            'Test.WITNESS_Eval.WITNESS.CCUS.carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix']
        pkl_name = f'jacobian_obj_vs_design_var_witness_coarse_subprocess.pkl'

        AbstractJacobianUnittest.DUMP_JACOBIAN = True
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
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process('climateeconomics.sos_processes.iam.witness',
                                                           'witness_optim_sub_process',
                                                           techno_dict=DEFAULT_COARSE_TECHNO_DICT,
                                                           invest_discipline=INVEST_DISCIPLINE_OPTIONS[2],
                                                           process_level='dev')
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(bspline=False,
                                           execution_engine=self.ee,
                                           techno_dict=DEFAULT_COARSE_TECHNO_DICT,
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
        # same hypothesis as uc1
        full_values_dict[
            f'{usecase.study_name}.{usecase.coupling_name}.{usecase.extra_name}.assumptions_dict'] = {
            'compute_gdp': False,
            'compute_climate_impact_on_gdp': False,
            'activate_climate_effect_population': False,
            'invest_co2_tax_in_renewables': False
        }
        full_values_dict[
            f"{usecase.study_name}.{usecase.coupling_name}.{usecase.extra_name}.ccs_price_percentage"] = 0.0
        full_values_dict[
            f"{usecase.study_name}.{usecase.coupling_name}.{usecase.extra_name}.co2_damage_price_percentage"] = 0.0
        self.ee.load_study_from_input_dict(full_values_dict)

        # Add design space to the study by filling design variables :
        design_space = pd.read_csv(join(dirname(__file__), 'data',  'all_iteration_dict.csv'))
        all_iterations_dspace_list = [eval(dspace) for dspace in design_space['value'].values]
        iter = 0
        test_results = []
        for dspace_dict in all_iterations_dspace_list[-1:]:
            self.ee.logger.info(f'testing iteration {iter}')
            design_space_values_dict = {}
            for variable_name, variable_value in dspace_dict.items():
                design_space_values_dict[self.ee.dm.get_all_namespaces_from_var_name(variable_name)[0]] = array(variable_value)

            self.ee.load_study_from_input_dict(design_space_values_dict)
            self.ee.execute()

            # loop over all disciplines

            coupling_disc = self.ee.root_process.proxy_disciplines[0]

            outputs = [self.ee.dm.get_all_namespaces_from_var_name(GlossaryCore.NegativeWelfareObjective)[0],
                       self.ee.dm.get_all_namespaces_from_var_name('objective_lagrangian')[0],
                       self.ee.dm.get_all_namespaces_from_var_name('Energy invest minimization objective')[0],
                       self.ee.dm.get_all_namespaces_from_var_name('last_year_discounted_utility_objective')[0],
                       self.ee.dm.get_all_namespaces_from_var_name('minimum_ppm_constraint')[0],
                       self.ee.dm.get_all_namespaces_from_var_name('Lower bound usable capital constraint')[0],
                       ]
            inputs_name = [f'{energy}_{techno}_array_mix' for energy, techno_dict in DEFAULT_COARSE_TECHNO_DICT.items() for
                           techno in techno_dict['value']]
            inputs_name = [name.replace('.', '_') for name in inputs_name]
            inputs = []
            for name in inputs_name:
                inputs.extend(self.ee.dm.get_all_namespaces_from_var_name(name))
            #inputs = [
            #    'Test.WITNESS_Eval.WITNESS.CCUS.carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix']
            pkl_name = f'jacobian_obj_vs_design_var_witness_coarse_subprocess_iter_{iter}.pkl'

            AbstractJacobianUnittest.DUMP_JACOBIAN = True
            try:
                self.check_jacobian(location=dirname(__file__), filename=pkl_name,
                                    discipline=coupling_disc.mdo_discipline_wrapp.mdo_discipline,
                                    step=1.0e-4, derr_approx='finite_differences', threshold=1e-15,
                                    local_data=coupling_disc.mdo_discipline_wrapp.mdo_discipline.local_data,
                                    inputs=inputs,
                                    outputs=outputs)
                test_results.append((iter, True))
                self.ee.logger.info(f'iteration {iter} succeeded')
            except AssertionError:
                test_results.append((iter, False))
                self.ee.logger.info(f'iteration {iter} failed')
            iter += 1
            self.ee.logger.info(f'Result of each iteration {test_results}')


