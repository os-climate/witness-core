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
import os
import pickle
import re
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
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_1_test_gradients_tp_3_5 import (
    Study as witness_optim_proc_uc1_tp_3_5,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_4_all_in_damage_high_tax import (
    Study as witness_optim_proc_uc4_tp_3_5,
)
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    Study as witness_sub_proc_usecase,
)
from climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline import (
    MacroeconomicsDiscipline,
)
from climateeconomics.sos_wrapping.sos_wrapping_witness.population.population_discipline import (
    PopulationDiscipline,
)

PKL_DM_REF_X = 'dm_uc1_tp_3_5_converged.pkl'
PKL_DM_REF_X_H = 'dm_uc1_X_h_tp_3_5_converged.pkl'
PKL_VAR_DIFF = 'uc1_tp_3_5_list_var_different.pkl'
PKL_COUPLED_VAR = 'uc1_tp_3_5_list_coupled_var.pkl'


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
        directory = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY)

        # recover the dm of the converged point of the mda (test_06) to update values_dict
        input_pkl = join(directory, PKL_DM_REF_X)
        with open(input_pkl, 'rb') as f:
            self.dm = pickle.load(f)
        f.close()

        input_pkl_h = join(directory, PKL_DM_REF_X_H)
        with open(input_pkl_h, 'rb') as f:
            self.dm_h = pickle.load(f)
        f.close()

        # recover the list of variables (at the converged point) that vary in X+h wrt to in X
        input_pkl = join(directory, PKL_VAR_DIFF)
        with open(input_pkl, 'rb') as f:
            self.list_chain_var_non_zero_gradient = pickle.load(f)
        f.close()

        # recover the list of the coupled variables in the mda (only ones to be considered for gradient computation)
        with open(join(directory, PKL_COUPLED_VAR), 'rb') as f:
            self.list_coupled_var_mda = pickle.load(f)
        f.close()


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

    def test_06_generate_ref_pkl(self):
        """
        Computes the full mda of witness coarse in a reference point X (=optimized point of usecase 1 at tipping point = 3.5°C)
        then in X + h.
        Then it saves the coupled variables and the variables of the mda that have a non-zero gradient, namely the variables which
        value differ in X and X+h
        """
        self.name = 'uc1_tp3_5'
        self.ee = ExecutionEngine(self.name)

        # if one invest discipline then we need to setup all subprocesses
        # before get them
        techno_dict = GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_coarse_dev_optim_process',
            techno_dict=techno_dict, invest_discipline=INVEST_DISCIPLINE_OPTIONS[2], process_level='dev', use_resources_bool=False)

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_optim_proc_uc1_tp_3_5(execution_engine=self.ee)

        usecase.study_name = self.name
        usecase.init_from_subusecase = True
        directory = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY)

        values_dict = usecase.setup_usecase()
        values_before = deepcopy(values_dict)
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        # export the data manager in X
        input_pkl = join(directory, PKL_DM_REF_X)
        dm = deepcopy(self.ee.dm.get_data_dict_values())
        with open(input_pkl, 'wb') as f:
            pickle.dump(dm, f)
        f.close()

        coupling_disc = self.ee.root_process.proxy_disciplines[0].proxy_disciplines[0]
        discipline = coupling_disc.mdo_discipline_wrapp.mdo_discipline

        # export all the coupled variables
        list_coupled_var = discipline.all_couplings
        with open(join(directory, PKL_COUPLED_VAR), 'wb') as f:
            pickle.dump(list_coupled_var, f)
        f.close()

        # compute the mda in X + h . Adjust the values_dict and the design_space
        h = 1.e-4
        for k, v in values_before.items():
            # should not consider the array_mix of forest in this case
            if 'utilization_ratio' in k or ('_array_mix' in k and 'forest' not in k) or 'share_non_energy_invest_ctrl' in k:
                values_before[k] += h
        design_space = values_before[f'{self.name}.WITNESS_MDO.design_space']
        design_space['value'] = design_space['value'].apply(lambda lst: [x + h for x in lst])

        self.ee.load_study_from_input_dict(values_before)
        self.ee.execute()

        dm_h = deepcopy(self.ee.dm.get_data_dict_values())
        input_pkl_h = join(directory, PKL_DM_REF_X_H)
        with open(input_pkl_h, 'wb') as f:
            pickle.dump(dm_h, f)
        f.close()

        # compare the dm to see what variables have changed in the X+h point
        with open(input_pkl, 'rb') as f:
            dm = pickle.load(f)
        f.close()
        #TODO: rtol in gemseo/utils/compare_data_manager_tooling.py l.188 must be checkexact or < 1.e-9
        compare_test_passed, error_msg_compare = test_compare_dm(dm, dm_h, self.name, 'dm in X vs in X+h')
        # extract the list of variables that differ
        pattern = r"Mismatch in \.(.*?)\:"
        list_var_different = re.findall(pattern, error_msg_compare)
        with open(join(directory, PKL_VAR_DIFF), 'wb') as f:
            pickle.dump(list_var_different, f)
        f.close()

    def test_06_gradient_process_objective2_over_design_var(self):
        """
        """
        self.name = 'uc1_tp3_5'
        self.ee = ExecutionEngine(self.name)

        # if one invest discipline then we need to setup all subprocesses
        # before get them
        techno_dict = GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_coarse_dev_optim_process',
            techno_dict=techno_dict, invest_discipline=INVEST_DISCIPLINE_OPTIONS[2], process_level='dev', use_resources_bool=False)

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_optim_proc_uc1_tp_3_5(execution_engine=self.ee)

        usecase.study_name = self.name
        usecase.init_from_subusecase = True
        directory = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY)

        values_dict = usecase.setup_usecase()
        values_before = deepcopy(values_dict)
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        # loop over all disciplines
        coupling_disc = self.ee.root_process.proxy_disciplines[0].proxy_disciplines[0]
        with open(os.path.join("data", "uc1optim.pkl"), "rb") as f:
            import pickle
            pickle.dump(self.ee.dm.get_data_dict_values(), f)
            pass
        discipline = coupling_disc.mdo_discipline_wrapp.mdo_discipline

        inputs = [
            self.ee.dm.get_all_namespaces_from_var_name('RenewableSimpleTechno.renewable_RenewableSimpleTechno_array_mix')[0], #OK lagr
            self.ee.dm.get_all_namespaces_from_var_name('FossilSimpleTechno.fossil_FossilSimpleTechno_array_mix')[0],
            ##self.ee.dm.get_all_namespaces_from_var_name('DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix')[0],
            ##self.ee.dm.get_all_namespaces_from_var_name('FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix')[0],
            ##self.ee.dm.get_all_namespaces_from_var_name('CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix')[0],
            self.ee.dm.get_all_namespaces_from_var_name('fossil_FossilSimpleTechno_utilization_ratio_array')[0],
            self.ee.dm.get_all_namespaces_from_var_name('renewable_RenewableSimpleTechno_utilization_ratio_array')[0], #OK lagr
            self.ee.dm.get_all_namespaces_from_var_name('share_non_energy_invest_ctrl')[0],
            ]
        #ref1 = f'_lagr_var_'
        ref1 = '_level_0_invest_mix_'
        #outputs = [self.ee.dm.get_all_namespaces_from_var_name('invest_mix')[0]]
        outputs = [self.ee.dm.get_all_namespaces_from_var_name('energy_production')[0],
                   self.ee.dm.get_all_namespaces_from_var_name('energy_mean_price')[0]
                   ]
        ref1 = '_level_2_ew_'
        """
        outputs = [self.ee.dm.get_all_namespaces_from_var_name('Quantity_objective')[0],
                   self.ee.dm.get_all_namespaces_from_var_name('decreasing_gdp_increments_obj')[0]
                   ]
        outputs = [self.ee.dm.get_all_namespaces_from_var_name('objective_lagrangian')[0]]
        ref1 = '_level_2_q_'
        ref1 = '_level_1_'
        outputs = [self.ee.dm.get_all_namespaces_from_var_name('population_df')[0],
                   self.ee.dm.get_all_namespaces_from_var_name('economics_detail_df')[0],
                   self.ee.dm.get_all_namespaces_from_var_name('detailed_capital_df')[0],
                   self.ee.dm.get_all_namespaces_from_var_name('economics_df')[0],
                   ]
        """
        location = dirname(__file__)
        filename = f'jacobian_lagrangian_{self.name}_vs_design_var.pkl'
        step = 1e-9
        derr_approx = 'finite_differences'
        threshold = 1.e-8
        override_dump_jacobian = True
        test_passed, dict_fail, dict_success = self.check_jac(discipline, inputs, outputs, location, filename, step,
                                                              derr_approx, threshold, override_dump_jacobian)

        assert test_passed

    def test_07_gradient_population(self):

        '''
        check_jacobian of population discipline at input conditions computed at converged mda of the test_06 chain
        '''
        self.name = 'Test_population'
        self.ee = ExecutionEngine(self.name)
        self.model_name = GlossaryCore.PopulationValue
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.population.population_discipline.PopulationDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        # Update inputs values
        list_input_var = list(PopulationDiscipline.DESC_IN.keys())
        # must complete the static inputs with the dynamic ones

        values_dict, list_variables_to_update_manually, values_dict_with_duplicates = self.create_values_dict_from_dm(list_input_var)

        # some variables are duplicate and need be adjusted manually
        print(f'Input variables to be updated manually = {list_variables_to_update_manually}')
        values_dict.update({
            f'{self.name}.year_start': 2020,
            f'{self.name}.theta': 2,
        })

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        # list inputs and outputs for the gradient computation
        #the default inputs checked in the l1 tests are:
        l1_inputs_checked = [f'{GlossaryCore.CaloriesPerCapitaValue}',
                             f'{GlossaryCore.EconomicsDfValue}',
                             f'{GlossaryCore.TemperatureDfValue}'
                             ]
        l1_outputs_checked = [f'{GlossaryCore.PopulationDfValue}',
                              f'{GlossaryCore.WorkingAgePopulationDfValue}',
                              ]


        inputs_var_grad, input_var_missing_in_l1_test = \
            self.determine_var_for_check_jac(list_input_var, l1_inputs_checked)
        if len(inputs_var_grad) < 1:
            raise ValueError(f'inputs for discipline {self.model_name} are only parameters. No gradient will be computed')

        # by default take all the disciplines' outputs.
        list_output_var = [var for var in list(PopulationDiscipline.DESC_OUT.keys())]
        # add the dynamic variables

        outputs_var_grad, output_var_missing_in_l1_test = \
            self.determine_var_for_check_jac(list_output_var, l1_outputs_checked)
        if len(outputs_var_grad) < 1:
            raise ValueError(
                f'outputs for discipline {self.model_name} are not varying in X+h. No gradient will be computed')

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        location = dirname(__file__)
        filename = f'jacobian_{self.name}_discipline_output.pkl'
        step = 1.e-4
        derr_approx = 'finite_differences'
        threshold = 1.e-8
        override_dump_jacobian = True

        test_passed, dict_fail, dict_success = self.check_jac(disc_techno, inputs_var_grad, outputs_var_grad, location, filename, step, derr_approx, threshold, override_dump_jacobian)

        assert test_passed

    def test_08_gradient_macroeconomics(self):

        '''
        check_jacobian of macroeconomics discipline at input conditions computed at converged mda of the test_06 chain
        '''
        self.name = 'Test_macroeconomics'
        self.ee = ExecutionEngine(self.name)
        self.model_name = 'Macroeconomics'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_MACRO: f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        # Update inputs values
        directory = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY)
        list_input_var_static = list(MacroeconomicsDiscipline.DESC_IN.keys())
        # some variables are named Macroeconomics.variable_name and must be renamed
        var_to_rename = [GlossaryCore.DamageToProductivity,
                         GlossaryCore.CO2TaxEfficiencyValue,
                         f'{GlossaryCore.SectorEnergyConsumptionPercentageDfName}_Services',
                         f'{GlossaryCore.SectionNonEnergyEmissionGdpDfValue}_Services',
                         f'{GlossaryCore.SectorEnergyConsumptionPercentageDfName}_Agriculture',
                         f'{GlossaryCore.SectionNonEnergyEmissionGdpDfValue}_Agriculture',
                         f'{GlossaryCore.SectorEnergyConsumptionPercentageDfName}_Industry',
                         f'{GlossaryCore.SectionNonEnergyEmissionGdpDfValue}_Industry',
        ]
        list_input_var = []
        for var in list_input_var_static:
            if var in var_to_rename:
                list_input_var.append(f'{self.model_name}.{var}')
            else:
                list_input_var.append(var)

        sector_list = self.dm['uc1_tp3_5.WITNESS_MDO.WITNESS_Eval.WITNESS.sector_list']
        for sector in sector_list:
            list_input_var.append(f'{GlossaryCore.SectorEnergyConsumptionPercentageDfName}_{sector}')
        list_input_var.append(GlossaryCore.SectionGdpPercentageDfValue)

        values_dict, list_variables_to_update_manually, values_dict_with_duplicates = self.create_values_dict_from_dm(
            list_input_var)
        # some variables are duplicate and need be adjusted manually
        print(f'Input variables to be updated manually = {list_variables_to_update_manually}')

        values_dict.update({
            f'{self.name}.sector_list': self.dm['uc1_tp3_5.WITNESS_MDO.WITNESS_Eval.WITNESS.sector_list'],
            f'{self.name}.energy_production': self.dm['uc1_tp3_5.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.energy_production'],
            f'{self.name}.sector_emission_consumption_percentage_df': self.dm['uc1_tp3_5.WITNESS_MDO.WITNESS_Eval.WITNESS.Macroeconomics.sector_emission_consumption_percentage_df'],
        })

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        # list inputs and outputs for the gradient computation
        #For macroeconomics, the default inputs and outputs checked in the l1 tests are:
        l1_inputs_checked = [f'{GlossaryCore.ShareNonEnergyInvestmentsValue}',
                               f'{GlossaryCore.EnergyProductionValue}',
                               f'{GlossaryCore.DamageFractionDfValue}',
                               f'{GlossaryCore.EnergyInvestmentsWoTaxValue}',
                               f'{GlossaryCore.CO2EmissionsGtValue}',
                               f'{GlossaryCore.CO2TaxesValue}',
                               f'{GlossaryCore.PopulationDfValue}',
                               f'{GlossaryCore.WorkingAgePopulationDfValue}',
                               f'{GlossaryCore.EnergyCapitalDfValue}',
                               ]
        l1_outputs_checked = [
            f'{self.model_name}.{GlossaryCore.TempOutput}',
            f'{GlossaryCore.DamageDfValue}',
            f'{GlossaryCore.EconomicsDfValue}',
            f'{GlossaryCore.EnergyInvestmentsValue}',
            f'{GlossaryCore.ConstraintLowerBoundUsableCapital}',
            f'{GlossaryCore.EnergyWastedObjective}',
            f'{GlossaryCore.ConsumptionObjective}',
            f'{GlossaryCore.UsableCapitalObjectiveName}'
        ]
        inputs_var_grad, input_var_missing_in_l1_test = \
            self.determine_var_for_check_jac(list_input_var, l1_inputs_checked)
        if len(inputs_var_grad) < 1:
            raise ValueError(
                f'inputs for discipline {self.model_name} are only parameters. No gradient will be computed')


        # by default take all the disciplines' outputs.
        list_output_var = [var for var in list(MacroeconomicsDiscipline.DESC_OUT.keys())]
        # add the dynamic variables
        for sector in sector_list:
            list_output_var.extend([f"{sector}.{GlossaryCore.SectionEnergyConsumptionDfValue}",
                                     f"{sector}.{GlossaryCore.SectionGdpDfValue}",
                                     ])

        list_output_var.extend([GlossaryCore.SectorGdpDfValue, GlossaryCore.AllSectionsGdpDfValue])

        outputs_var_grad, output_var_missing_in_l1_test = \
            self.determine_var_for_check_jac(list_output_var, l1_outputs_checked)
        if len(outputs_var_grad) < 1:
            raise ValueError(
                f'outputs for discipline {self.model_name} are not varying in X+h. No gradient will be computed')

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        location = dirname(__file__)
        filename = f'jacobian_{self.name}_discipline_output.pkl'
        step = 1.e-4
        derr_approx = 'finite_differences'
        threshold = 1.e-8
        override_dump_jacobian = True
        test_passed, dict_fail, dict_success = self.check_jac(disc_techno, inputs_var_grad, outputs_var_grad, location, filename, step, derr_approx, threshold, override_dump_jacobian)
        assert test_passed

    def test_09_gradient_energy_mix(self):

        '''
        check_jacobian of energy_mix discipline at input conditions computed at converged mda of the test_06 chain
        '''
        self.name = 'uc1_tp3_5'  # must have the same name as test_06 in order to compare diredctly the variables names with the coupled variables
        self.ee = ExecutionEngine(self.name)
        self.model_name = 'EnergyMix'
        # if one invest discipline then we need to setup all subprocesses
        # before get them
        techno_dict = GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_coarse_dev_optim_process',
            techno_dict=techno_dict, invest_discipline=INVEST_DISCIPLINE_OPTIONS[2], process_level='dev', use_resources_bool=False)

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_optim_proc_uc1_tp_3_5(execution_engine=self.ee)

        usecase.study_name = self.name
        usecase.init_from_subusecase = True

        values_dict = usecase.setup_usecase()

        self.ee.display_treeview_nodes()

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        disc_techno = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix')[0].mdo_discipline_wrapp.mdo_discipline

        list_input_var_all = disc_techno.input_grammar.data_names
        list_output_var_all = disc_techno.output_grammar.data_names
        # only keep the coupled in inputs and coupled outputs. Must remove the self.name at the beginning
        list_input_var = [var.split(f'{self.name}.')[1] for var in list_input_var_all if var in self.list_coupled_var_mda]
        list_output_var = [var.split(f'{self.name}.')[1] for var in list_output_var_all if var in self.list_coupled_var_mda]


        # list inputs and outputs for the gradient computation
        #the default inputs and outputs checked in the l1 tests are:
        l1_inputs_checked = []
        energy_list = [GlossaryEnergy.renewable, GlossaryEnergy.fossil]

        l1_inputs_checked.extend([
            f'WITNESS_MDO.WITNESS_Eval.WITNESS.{self.model_name}.{energy}.{GlossaryEnergy.StreamPricesValue}' for energy in energy_list
            if
            energy not in [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])
        l1_inputs_checked.extend([
            f'WITNESS_MDO.WITNESS_Eval.WITNESS.{self.model_name}.{energy}.{GlossaryEnergy.EnergyProductionValue}' for energy in
            energy_list if
            energy not in [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])
        l1_inputs_checked.extend(
            [f'WITNESS_MDO.WITNESS_Eval.WITNESS.{self.model_name}.{energy}.{GlossaryEnergy.EnergyConsumptionValue}' for energy in
             energy_list if
             energy not in [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])
        l1_inputs_checked.extend(
            [f'WITNESS_MDO.WITNESS_Eval.WITNESS.{GlossaryEnergy.ccus_type}.{energy}.{GlossaryEnergy.EnergyConsumptionValue}' for energy in
             [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])

        l1_inputs_checked.extend(
            [f'WITNESS_MDO.WITNESS_Eval.WITNESS.{GlossaryEnergy.ccus_type}.{energy}.{GlossaryEnergy.EnergyProductionValue}' for energy in
             [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])
        l1_inputs_checked.extend([
            f'WITNESS_MDO.WITNESS_Eval.WITNESS.{GlossaryEnergy.ccus_type}.{energy}.{GlossaryEnergy.StreamPricesValue}' for energy in
            [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])
        l1_inputs_checked.extend(
            [f'WITNESS_MDO.WITNESS_Eval.WITNESS.{self.model_name}.{energy}.{GlossaryEnergy.CO2EmissionsValue}' for energy in energy_list
             if
             energy not in [GlossaryEnergy.carbon_capture, GlossaryEnergy.carbon_storage]])


        l1_outputs_checked = [f'WITNESS_MDO.WITNESS_Eval.WITNESS.{self.model_name}.{GlossaryEnergy.EnergyMeanPriceValue}',
                             f'WITNESS_MDO.WITNESS_Eval.WITNESS.FunctionManagerDisc.{GlossaryEnergy.EnergyMeanPriceObjectiveValue}',
                             f'WITNESS_MDO.WITNESS_Eval.WITNESS.{self.model_name}.energy_prices_after_tax',
                              f'WITNESS_MDO.WITNESS_Eval.WITNESS.{self.model_name}.co2_emissions_needed_by_energy_mix',
                              f'WITNESS_MDO.WITNESS_Eval.WITNESS.{self.model_name}.carbon_capture_from_energy_mix',
                              f'WITNESS_MDO.WITNESS_Eval.WITNESS.{self.model_name}.{GlossaryEnergy.EnergyMeanPriceValue}',
                              f'WITNESS_MDO.WITNESS_Eval.WITNESS.{self.model_name}.{GlossaryEnergy.EnergyProductionValue}',
                              f'WITNESS_MDO.WITNESS_Eval.WITNESS.{self.model_name}.land_demand_df',
                              f'WITNESS_MDO.WITNESS_Eval.WITNESS.{GlossaryEnergy.EnergyCapitalDfValue}',
                              f'WITNESS_MDO.WITNESS_Eval.WITNESS.{self.model_name}.energy_prices_after_tax',
                              f'FunctionManagerDisc.{GlossaryEnergy.TargetProductionConstraintValue}',
                              ]

        inputs_var_grad, input_var_missing_in_l1_test = \
            self.determine_var_for_check_jac(list_input_var, l1_inputs_checked)
        if len(inputs_var_grad) < 1:
            raise ValueError(
                f'inputs for discipline {self.model_name} are only parameters. No gradient will be computed')

        outputs_var_grad, output_var_missing_in_l1_test = \
            self.determine_var_for_check_jac(list_output_var, l1_outputs_checked)
        if len(outputs_var_grad) < 1:
            raise ValueError(
                f'outputs for discipline {self.model_name} are not varying in X+h. No gradient will be computed')

        # no need to check all gradients since the gradient error in the lagrangian only comes from the energy_mean_price
        outputs_var_grad = [f'{self.name}.WITNESS_MDO.WITNESS_Eval.WITNESS.{self.model_name}.{GlossaryEnergy.EnergyMeanPriceValue}']

        location = dirname(__file__)
        filename = f'jacobian_{self.name}_discipline_output.pkl'
        step = 1.e-4
        derr_approx = 'finite_differences'
        threshold = 1.e-8
        override_dump_jacobian = True
        test_passed, dict_fail, dict_success = self.check_jac(disc_techno, inputs_var_grad, outputs_var_grad, location, filename, step, derr_approx, threshold, override_dump_jacobian)
        assert test_passed

    def test_10_compare_dm(self):
        '''
        test if the equal method of a series and the test_compare_dm provide the same
        result for 2 series that differ by 1.e-4 on all elements
        '''
        # isolate variable that is problematic
        dm0 = {'var1': self.dm['uc1_tp3_5.WITNESS_MDO.WITNESS_Eval.WITNESS.share_non_energy_investment']}
        dm0_h = {'var1': self.dm_h['uc1_tp3_5.WITNESS_MDO.WITNESS_Eval.WITNESS.share_non_energy_investment']}

        df_are_equal = (dm0['var1']['share_non_energy_investment'] == dm0_h['var1']['share_non_energy_investment']).all()
        # TODO change rtol from 1e-3 to 1e-9 at least or check abs
        compare_test_passed, error_msg_compare = test_compare_dm(dm0, dm0_h, self.name, 'dm in X vs in X+h')

        assert compare_test_passed==df_are_equal

    def check_jac(self, discipline, inputs, outputs, location, filename, step, derr_approx, threshold, override_dump_jacobian):
        test_pass = True
        dict_success = {}
        dict_fail = {}
        for j, output in enumerate(outputs):
            dict_success[output] = []
            dict_fail[output] = []
            for i, inpt in enumerate(inputs):
                ref = f'_input_{i}_output_{j}'
                pkl_name = f"{filename.split('.pkl')[0]}{ref}.pkl"
                self.override_dump_jacobian = override_dump_jacobian
                try:
                    # TODO: correct if condition of 597 of gemseo/mda/mda.py
                    self.check_jacobian(location=location, filename=pkl_name,
                                        discipline=discipline, local_data=discipline.local_data,
                                        inputs=[inpt], outputs=[output], step=step, derr_approx=derr_approx,
                                        threshold=threshold)
                    dict_success[output].append(inpt)
                    os.remove("jacobian_pkls", pkl_name)
                except Exception as e:
                    print(e)
                    dict_fail[output].append(inpt)
                    test_pass = False

        for k, v in dict_fail.items():
            print(f'## FAIL FOR OUTPUT {k}, INPUTS: {v}')
        for k, v in dict_success.items():
            print(f'## SUCCESS FOR OUTPUT {k}, INPUTS: {v}')

        return test_pass, dict_fail, dict_success

    def create_values_dict_from_dm(self, inputs):
        '''
        create a dictionary k, v where k=discipline variable names and v=value taken from a data_manager previously computed
        in another study and loaded under self.dm_ref
        Args:
            inputs: [list] list of the discipline's generic input variable name (without the namespace value added)
        '''
        values_dict_with_duplicates = {k: [] for k in inputs}
        for key, val in self.dm.items():
            for k, v in values_dict_with_duplicates.items():
                # key is of format namespace_value.variable_name.
                # Add the . in the check to avoid working_age_population_df to come up for instance when checking population_df
                if f'.{k}' in key:
                    values_dict_with_duplicates[k].append(val)

        values_dict = {f'{self.name}.{k}': v[0] for k, v in values_dict_with_duplicates.items()}

        # determine the variables with duplicates which values must be updated manually
        variables_to_update_manually = []
        for k, v in values_dict_with_duplicates.items():
            if len(v) > 1:
                variables_to_update_manually.append(f'{self.name}.{k}')

        return values_dict, variables_to_update_manually, values_dict_with_duplicates

    def determine_var_for_check_jac(self, discipline_var, l1_test_var):
        '''
        Determine the discipline inputs or outputs to consider for the check_jacobian: they are those that
           -  vary when the full chain gradient is computed wrt the design variables. those variables are listed in the input_pkl
           - and that are coupled in the mda chain
        Args:
            discipline_var: [list] list of the generic input or output variable names of the discipline
            l1_test_var: [list] list of the generic input or output variable names of the discipline that are used in the l1 test to
                        check the jacobian
        Returns:
            var_grad: [list] list of variables to consider to compute the gradient of the discipline
            var_missing_in_l1_test: [list] list of variables that have been forgotten in the l1_test
        '''
        var_grad = []
        # filter out the parameters
        for x in discipline_var:
            # only variables that are weakly coupled in the mda chain and that vary in X+h wrt value in X need their grad to be evaluated
            for var in list(set(self.list_coupled_var_mda).intersection(set(self.list_chain_var_non_zero_gradient))):
                if (x in var) and (f'{self.name}.{x}' not in var_grad): # avoid duplicates if var exist in different disciplines
                    var_grad.append(f'{self.name}.{x}')

        # reconstruct the full name of the variable including the test name in order to compare var with var_grad
        l1_test_var_full_name = [f'{self.name}.{var}' for var in l1_test_var]
        var_missing_in_l1_test = [var for var in var_grad if var not in l1_test_var_full_name]

        return var_grad, var_missing_in_l1_test

    def test_111_gradient_process_objective2_over_design_var(self):
        """
        """
        self.name = 'uc4_tp3_5'
        self.ee = ExecutionEngine(self.name)

        # if one invest discipline then we need to setup all subprocesses
        # before get them
        techno_dict = GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT

        builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_coarse_dev_optim_process',
            techno_dict=techno_dict, invest_discipline=INVEST_DISCIPLINE_OPTIONS[2], process_level='dev', use_resources_bool=False)

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = witness_optim_proc_uc4_tp_3_5(execution_engine=self.ee)

        usecase.study_name = self.name
        usecase.init_from_subusecase = True
        directory = join(dirname(__file__), AbstractJacobianUnittest.PICKLE_DIRECTORY)

        values_dict = usecase.setup_usecase()
        values_before = deepcopy(values_dict)
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        # loop over all disciplines
        coupling_disc = self.ee.root_process.proxy_disciplines[0].proxy_disciplines[0]
        with open(os.path.join("data","uc4optim.pkl"), "wb") as f:
            import pickle
            pickle.dump(self.ee.dm.get_data_dict_values(), f)
            pass
        discipline = coupling_disc.mdo_discipline_wrapp.mdo_discipline
        """
        inputs = [
            #self.ee.dm.get_all_namespaces_from_var_name('RenewableSimpleTechno.renewable_RenewableSimpleTechno_array_mix')[0], #OK lagr
            #self.ee.dm.get_all_namespaces_from_var_name('FossilSimpleTechno.fossil_FossilSimpleTechno_array_mix')[0],
            self.ee.dm.get_all_namespaces_from_var_name('DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix')[0],
            #self.ee.dm.get_all_namespaces_from_var_name('FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix')[0],
            #self.ee.dm.get_all_namespaces_from_var_name('CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix')[0],
            #self.ee.dm.get_all_namespaces_from_var_name('fossil_FossilSimpleTechno_utilization_ratio_array')[0],
            #self.ee.dm.get_all_namespaces_from_var_name('renewable_RenewableSimpleTechno_utilization_ratio_array')[0], #OK lagr
            #self.ee.dm.get_all_namespaces_from_var_name('carbon_capture.direct_air_capture.DirectAirCaptureTechno_utilization_ratio_array')[0],
            #self.ee.dm.get_all_namespaces_from_var_name('carbon_capture.flue_gas_capture.FlueGasTechno_utilization_ratio_array')[0],
            #self.ee.dm.get_all_namespaces_from_var_name('carbon_storage.CarbonStorageTechno_utilization_ratio_array')[0],
            #self.ee.dm.get_all_namespaces_from_var_name('share_non_energy_invest_ctrl')[0],
            ]
        outputs = [
            #self.ee.dm.get_all_namespaces_from_var_name('objective_lagrangian')[0],
            self.ee.dm.get_all_namespaces_from_var_name(
                f'carbon_capture.{GlossaryEnergy.direct_air_capture}.{GlossaryEnergy.DirectAirCaptureTechno}.invest_level')[
                0], # OK
            self.ee.dm.get_all_namespaces_from_var_name(
                f'carbon_capture.{GlossaryEnergy.direct_air_capture}.{GlossaryEnergy.DirectAirCaptureTechno}.{GlossaryEnergy.TechnoProductionValue}')[
                0], # OK
            self.ee.dm.get_all_namespaces_from_var_name(
                f'carbon_capture.{GlossaryEnergy.direct_air_capture}.{GlossaryEnergy.DirectAirCaptureTechno}.techno_prices')[
                0], # NOK
            self.ee.dm.get_all_namespaces_from_var_name(
                f'carbon_capture.{GlossaryEnergy.direct_air_capture}.{GlossaryEnergy.DirectAirCaptureTechno}.{GlossaryEnergy.TechnoConsumptionValue}')[
                0],
            self.ee.dm.get_all_namespaces_from_var_name(
                f'carbon_capture.{GlossaryEnergy.direct_air_capture}.{GlossaryEnergy.DirectAirCaptureTechno}.{GlossaryEnergy.CO2EmissionsValue}')[
                0],
            #self.ee.dm.get_all_namespaces_from_var_name('carbon_capture.energy_production')[0],
            #self.ee.dm.get_all_namespaces_from_var_name('carbon_capture.energy_prices')[0],

            #self.ee.dm.get_all_namespaces_from_var_name('energy_production')[0],
            #self.ee.dm.get_all_namespaces_from_var_name('energy_mean_price')[0],
        ]
        #ref1 = f'_lagr_var_'
        
        outputs = [self.ee.dm.get_all_namespaces_from_var_name('objective_lagrangian')[0]]
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
        outputs = [self.ee.dm.get_all_namespaces_from_var_name('objective_lagrangian')[0]]
        

        location = dirname(__file__)
        filename = f'jacobian_lagrangian_{self.name}_vs_design_var.pkl'
        step = 1e-9
        derr_approx = 'finite_differences'
        threshold = 1.e-5
        override_dump_jacobian = True
        test_passed, dict_fail, dict_success = self.check_jac(discipline, inputs, outputs, location, filename, step,
                                                              derr_approx, threshold, override_dump_jacobian)

        assert test_passed
        """
