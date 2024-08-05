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
import pickle
from copy import deepcopy
from os.path import dirname, join

import numpy as np
import pandas as pd
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.glossaryenergy import GlossaryEnergy
from numpy import array
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.sos_processes.script_test_all_usecases import test_compare_dm
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)
from sostrades_optimization_plugins.models.func_manager.func_manager import (
    FunctionManager,
)
from sostrades_optimization_plugins.models.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_1_fossil_only_no_damage_low_tax import (
    Study as witness_optim_proc_uc1,
)
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    OPTIM_NAME,
)
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    Study as witness_sub_proc_usecase,
)


class OptimSubprocessJacobianDiscTest(AbstractJacobianUnittest):

    '''
    WARNING !!!!!!:
    there is a bug in gemseo that outputs a check_jacobian = True eventhough the output variables chosen are
    coupled variables (hence gemseo outputs an approximated jacobian = {}). To cope with this:
    - either only chose output variables = uncoupled variables (=> strongly limit the tests)
    - or comment the if condition at line 597 of gemseo/mda/mda.py so that the approximated gradient is outputed
    '''

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
            f'Test.WITNESS_Eval.WITNESS.CCUS.carbon_capture.direct_air_capture.{GlossaryEnergy.DirectAirCaptureTechno}.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix']
        pkl_name = 'jacobian_obj_vs_design_var_witness_coarse_subprocess.pkl'

        # self.override_dump_jacobian = True
        #TODO: correct if condition of 597 of gemseo/mda/mda.py
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
            #    'Test.WITNESS_Eval.WITNESS.CCUS.carbon_capture.direct_air_capture.{GlossaryEnergy.DirectAirCaptureTechno}.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix']
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
                # TODO: correct if condition of 597 of gemseo/mda/mda.py
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

        mod_list = 'sostrades_optimization_plugins.models.func_manager.func_manager_disc.FunctionManagerDisc'
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

    def test_04_gradient_subprocess_objective2_over_design_var(self):
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
        full_values_dict[
            f'{usecase.study_name}.{usecase.coupling_name}.{usecase.extra_name}.Damage.tp_a3'] = 3.5
        full_values_dict[
            f'{usecase.study_name}.{usecase.coupling_name}.{usecase.extra_name}.share_non_energy_invest_ctrl'] = np.array([27.0] * (GlossaryCore.NB_POLES_COARSE-1))

        # get design space and dvar descriptor from climateeconomics/sos_processes/iam/witness/witness_coarse_dev_optim_process/usecase_1_fossil_only_no_damage_low_tax.py
        # with converged point at tipping point = 3.5 deg from the optim process
        input_design_space_pkl = join(dirname(__file__), 'design_space_uc1_tp_3_5.pkl')
        input_dv_descriptor_pkl = join(dirname(__file__), 'dv_descriptor_uc1_tp_3_5.pkl')
        with open(input_design_space_pkl, 'rb') as f:
            dspace = pickle.load(f)
        f.close()
        with open(input_dv_descriptor_pkl, 'rb') as f:
            dv_descriptor = pickle.load(f)
        f.close()

        # activate all elements otherwise check grad collapses for the design variables discipline
        for i in range(len(dspace['activated_elem'])):
            dspace['activated_elem'].iloc[i][0] = True
        usecase.dspace = dspace
        updated_data = {
            f'{usecase.study_name}.{usecase.coupling_name}.DesignVariables.design_var_descriptor': dv_descriptor,
            f'{usecase.study_name}.design_space': dspace,
        }
        full_values_dict.update(updated_data)
        # must load the study to have access to the variables
        self.ee.load_study_from_input_dict(full_values_dict)

        design_space_values_dict = {}
        for variable in dspace['variable'].values:
            val = dspace[dspace['variable'] == variable]['value'].values[0]
            design_space_values_dict[self.ee.dm.get_all_namespaces_from_var_name(variable)[0]] = array(val)

        design_space_values_dict.update(updated_data)
        self.ee.load_study_from_input_dict(design_space_values_dict)

        self.ee.execute()

        # loop over all disciplines

        coupling_disc = self.ee.root_process.proxy_disciplines[0]

        inputs = [
            #'Test.WITNESS_Eval.WITNESS.EnergyMix.renewable.RenewableSimpleTechno.renewable_RenewableSimpleTechno_array_mix', #OK lagr
            'Test.WITNESS_Eval.WITNESS.EnergyMix.fossil.FossilSimpleTechno.fossil_FossilSimpleTechno_array_mix',
            ##'Test.WITNESS_Eval.WITNESS.CCUS.carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix',
            ##'Test.WITNESS_Eval.WITNESS.CCUS.carbon_capture.flue_gas_capture.FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix',
            ##'Test.WITNESS_Eval.WITNESS.CCUS.carbon_storage.CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix',
            'Test.WITNESS_Eval.WITNESS.EnergyMix.fossil_FossilSimpleTechno_utilization_ratio_array',
            #'Test.WITNESS_Eval.WITNESS.EnergyMix.renewable_RenewableSimpleTechno_utilization_ratio_array', #OK lagr
            'Test.WITNESS_Eval.WITNESS.share_non_energy_invest_ctrl'
            ]
        ref1 = '_level_lagr_var_'
        outputs = [self.ee.dm.get_all_namespaces_from_var_name('objective_lagrangian')[0]]
        #ref = f'_level_0_invest_mix_'
        #outputs = [self.ee.dm.get_all_namespaces_from_var_name('invest_mix')[0]]
        #ref = f'_level_1_'
        outputs = [self.ee.dm.get_all_namespaces_from_var_name('energy_wasted_objective')[0],
                   #self.ee.dm.get_all_namespaces_from_var_name('Quantity_objective')[0],
                   #self.ee.dm.get_all_namespaces_from_var_name('decreasing_gdp_increments_obj')[0],
                   ]
        dict_success = {}
        dict_fail = {}
        for j, output in enumerate(outputs):
            dict_success[output] = []
            dict_fail[output] = []
            for i, input in enumerate(inputs):
                print('############')
                print(f'output = {output}, input={input}')
                ref2 = f'_{i}_{j}'

                pkl_name = 'jacobian_obj_vs_design_var_witness_coarse_subprocess' + ref1 + ref2 + 'fd4_.pkl'

                self.override_dump_jacobian = True
                try:
                    # TODO: correct if condition of 597 of gemseo/mda/mda.py
                    self.check_jacobian(location=dirname(__file__), filename=pkl_name,
                                        discipline=coupling_disc.mdo_discipline_wrapp.mdo_discipline,
                                        step=1.0e-4, derr_approx='finite_differences', threshold=1e-8,
                                        local_data=coupling_disc.mdo_discipline_wrapp.mdo_discipline.local_data,
                                        inputs=[input],
                                        outputs=[output])
                    dict_success[output].append(input)
                except:
                    dict_fail[output].append(input)
        for k, v in dict_fail.items():
            print(f'## FAIL FOR OUTPUT {k}, INPUTS: {v}')
        for k, v in dict_success.items():
            print(f'## SUCCESS FOR OUTPUT {k}, INPUTS: {v}')

    def test_05_jacobian_func_manager_disc(self):
        '''
        test_04 showed that the gradient of the sub-objectives was OK (wrt design var) whereas it was not for the lagrangian
        => isolate the lagrangian gradient wrt sub-objective variables at the same point as the one studied in
        test_04.
        Test passes for threshold > 1.e-12 and step = 1.e-4
        Test crashes for threshold = 1.e-12 and step = 1.e-4
        '''
        self.func_manager = FunctionManager()
        OBJECTIVE = self.func_manager.OBJECTIVE
        OBJECTIVE_LAGR = FunctionManagerDisc.OBJECTIVE_LAGR

        # -- init the case
        func_mng_name = 'FunctionManager'
        prefix = self.name + '.' + func_mng_name + '.'

        ee = ExecutionEngine(self.name)
        ns_dict = {'ns_functions': self.name + '.' + func_mng_name,
                   'ns_optim': self.name + '.' + func_mng_name}
        ee.ns_manager.add_ns_def(ns_dict)

        mod_list = 'sostrades_optimization_plugins.models.func_manager.func_manager_disc.FunctionManagerDisc'
        fm_builder = ee.factory.get_builder_from_module(
            'FunctionManager', mod_list)
        ee.factory.set_builders_to_coupling_builder(fm_builder)
        ee.configure()

        # -- i/o setup
        '''
        WARNING: in the optim, the sub-objective functions are 1D variable array-like as defined as output of disciplines 
        utility and macroeconomics. However, here, the sub-objective functions are defined from scratch and therefore, 
        sostrades_optimization_plugins/models/func_manager/func_manager_disc.py, will execpt a dataframe .
        Therefore, to make a test consistent with what happens in the optimization, line 213 of 
        sostrades_optimization_plugins/models/func_manager/func_manager_disc.py 
        must be changed by self.TYPE: 'array' instead of self.TYPE: 'dataframe'
        '''
        obj_energy_wasted = array([0.34908776])
        obj_quantity = array([0.84998263])
        obj_decreasing_gdp = array([-0.])

        # -- ~GUI inputs: selection of functions

        func_df = pd.DataFrame(columns=['variable', 'ftype', 'weight'])
        func_df['variable'] = ['obj_energy_wasted', 'obj_quantity', 'obj_decreasing_gdp',
                               ]
        func_df['ftype'] = [OBJECTIVE, OBJECTIVE, OBJECTIVE]
        func_df['weight'] = [0.1, -1., 1.]
        func_df['aggr'] = ["sum", "sum", "sum"]
        func_df['parent'] = ['invest_objective', 'utility_objective', 'utility_objective']
        func_df['namespace'] = ['ns_functions', 'ns_functions', 'ns_functions']


        values_dict = {}
        values_dict[prefix + FunctionManagerDisc.FUNC_DF] = func_df

        # -- data to simulate disciplinary chain outputs
        values_dict[prefix + 'obj_energy_wasted'] = obj_energy_wasted
        values_dict[prefix + 'obj_quantity'] = obj_quantity
        values_dict[prefix + 'obj_decreasing_gdp'] = obj_decreasing_gdp

        ee.load_study_from_input_dict(values_dict)

        ee.display_treeview_nodes(True)

        # -- execution
        ee.execute()
        # -- retrieve outputs
        disc = ee.dm.get_disciplines_with_name(
            f'{self.name}.{func_mng_name}')[0]
        outputs = disc.get_sosdisc_outputs()

        # -- check outputs with reference data
        self.assertAlmostEqual(outputs[OBJECTIVE][0], (0.1 * obj_energy_wasted - 1. * obj_quantity + 1. * obj_decreasing_gdp)[0])

        res = 100. * (outputs[OBJECTIVE][0])

        self.assertEqual(outputs[OBJECTIVE_LAGR][0], res)

        disc_techno = ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        assert disc_techno.check_jacobian(
            input_data=disc_techno.local_data,
            step=1.e-4,
            threshold=1e-11,
            inputs=['Test.FunctionManager.obj_energy_wasted',
                                    'Test.FunctionManager.obj_quantity',
                                    'Test.FunctionManager.obj_quantity'],
            outputs=['Test.FunctionManager.objective_lagrangian'], derr_approx='finite_differences')

    def test_06_gradient_process_objective2_over_design_var(self):
        """
        """
        self.name = 'usecase_1_fossil_only_no_damage_low_tax'
        self.ee = ExecutionEngine(self.name)

        optim_name = OPTIM_NAME

        # if one invest discipline then we need to setup all subprocesses
        # before get them
        techno_dict = GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_coarse_dev_optim_process',
            techno_dict=techno_dict, invest_discipline=INVEST_DISCIPLINE_OPTIONS[2], process_level='dev', use_resources_bool=False)

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_optim_proc_uc1(execution_engine=self.ee)

        usecase.study_name = self.name
        usecase.init_from_subusecase = True
        directory = join(AbstractJacobianUnittest.PICKLE_DIRECTORY, 'optim_check_gradient_dev')

        values_dict = usecase.setup_usecase()
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        # Compare dm with usecase 1
        dm = deepcopy(self.ee.dm.get_data_dict_values())
        with open(r'C:\Users\bherry\Documents\dm_comp.pkl', 'rb') as f:
            dm0 = pickle.load(f)
        f.close()
        compare_test_passed, error_msg_compare = test_compare_dm(dm, dm0, 'RAS', 'pkl vs case dm')
        # loop over all disciplines

        coupling_disc = self.ee.root_process.proxy_disciplines[0].proxy_disciplines[0]

        inputs = [
            #self.ee.dm.get_all_namespaces_from_var_name('RenewableSimpleTechno.renewable_RenewableSimpleTechno_array_mix')[0], #OK lagr
            self.ee.dm.get_all_namespaces_from_var_name('FossilSimpleTechno.fossil_FossilSimpleTechno_array_mix')[0],
            ##self.ee.dm.get_all_namespaces_from_var_name('DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix')[0],
            ##self.ee.dm.get_all_namespaces_from_var_name('FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix')[0],
            ##self.ee.dm.get_all_namespaces_from_var_name('CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix')[0],
            self.ee.dm.get_all_namespaces_from_var_name('fossil_FossilSimpleTechno_utilization_ratio_array')[0],
            #self.ee.dm.get_all_namespaces_from_var_name('renewable_RenewableSimpleTechno_utilization_ratio_array')[0], #OK lagr
            self.ee.dm.get_all_namespaces_from_var_name('share_non_energy_invest_ctrl')[0],
            ]
        #ref1 = f'_lagr_var_'
        #outputs = [self.ee.dm.get_all_namespaces_from_var_name('objective_lagrangian')[0]]
        ref1 = '_level_0_invest_mix_'
        outputs = [self.ee.dm.get_all_namespaces_from_var_name('invest_mix')[0]]
        ref1 = '_level_1_'
        outputs = [self.ee.dm.get_all_namespaces_from_var_name('energy_wasted_objective')[0],
                   self.ee.dm.get_all_namespaces_from_var_name('Quantity_objective')[0],
                   self.ee.dm.get_all_namespaces_from_var_name('decreasing_gdp_increments_obj')[0]
                   ]
        ref1 = '_level_2_ew_'
        outputs = [self.ee.dm.get_all_namespaces_from_var_name('energy_production')[0],
                   self.ee.dm.get_all_namespaces_from_var_name('detailed_capital_df')[0],
                   ]
        ref1 = '_level_2_q_'
        outputs = [self.ee.dm.get_all_namespaces_from_var_name('population_df')[0],
                   self.ee.dm.get_all_namespaces_from_var_name('economics_detail_df')[0],
                   self.ee.dm.get_all_namespaces_from_var_name('economics_df')[0],
                   self.ee.dm.get_all_namespaces_from_var_name('energy_mean_price')[0]
                   ]
        dict_success = {}
        dict_fail = {}
        for j, output in enumerate(outputs):
            dict_success[output] = []
            dict_fail[output] = []
            for i, input in enumerate(inputs):
                print('############')
                print(f'output = {output}, input={input}')
                ref2 = f'_{i}_{j}'

                pkl_name = 'jacobian_obj_vs_design_var_witness_coarse_process' + ref1 + ref2 + '.pkl'

                self.override_dump_jacobian = True
                try:
                    # TODO: correct if condition of 597 of gemseo/mda/mda.py
                    self.check_jacobian(location=dirname(__file__), filename=pkl_name,
                                        discipline=coupling_disc.mdo_discipline_wrapp.mdo_discipline,
                                        step=1.0e-4, derr_approx='finite_differences', threshold=1e-8,
                                        local_data=coupling_disc.mdo_discipline_wrapp.mdo_discipline.local_data,
                                        inputs=[input],
                                        outputs=[output])
                    dict_success[output].append(input)
                except:
                    dict_fail[output].append(input)
        for k, v in dict_fail.items():
            print(f'## FAIL FOR OUTPUT {k}, INPUTS: {v}')
        for k, v in dict_success.items():
            print(f'## SUCCESS FOR OUTPUT {k}, INPUTS: {v}')
        print()


