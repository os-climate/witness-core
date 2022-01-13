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
"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
"""

import numpy as np
from os.path import join, dirname

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest
import pickle


class CoarseJacobianTestCase(AbstractJacobianUnittest):
    """
    Ratio jacobian test class
    """
    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def analytic_grad_entry(self):
        return [
            self.test_01_coarse_renewable_techno_discipline_jacobian,
            self.test_02_coarse_fossil_techno_discipline_jacobian,
            self.test_03_coarse_dac_techno_discipline_jacobian,
            self.test_04_coarse_flue_gas_capture_techno_discipline_jacobian,
            self.test_05_coarse_carbon_storage_techno_discipline_jacobian,
            self.test_06_coarse_renewable_stream_discipline_jacobian,
            self.test_07_coarse_fossil_stream_discipline_jacobian,
        ]

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'Test_Coarse'
        years = np.arange(2020, 2051)
        self.years = years

    def tearDown(self):
        pass

    def test_01_coarse_renewable_techno_discipline_jacobian(self):
        '''
        Test the gradients of coarse renewable techno
        '''
        self.techno_name = 'RenewableSimpleTechno'
        self.ee = ExecutionEngine(self.name)
        ns_dict = {'ns_public': self.name, 'ns_energy': self.name,
                   'ns_energy_study': f'{self.name}',
                   'ns_renewable': f'{self.name}',
                   'ns_resource': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'energy_models.models.renewable.renewable_simple_techno.renewable_simple_techno_disc.RenewableSimpleTechnoDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.techno_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        pkl_file = open(
            join(dirname(__file__), 'data/mda_coarse_data_technologies_input_dict.pkl'), 'rb')
        mda_data_input_dict = pickle.load(pkl_file)
        pkl_file.close()

        namespace = f'{self.name}'
        inputs_dict = {}
        coupled_inputs = []
        for key in mda_data_input_dict[self.techno_name].keys():
            # Modify namespace of input 'key' if needed
            if key in ['linearization_mode', 'cache_type', 'cache_file_path', 'sub_mda_class',
                       'max_mda_iter', 'n_processes', 'chain_linearize', 'tolerance', 'use_lu_fact',
                       'warm_start', 'acceleration', 'warm_start_threshold', 'n_subcouplings_parallel',
                       'max_mda_iter_gs', 'condition_func', 'relax_factor', 'epsilon0', 'reset_history_each_run',
                       'linear_solver_MDO', 'linear_solver_MDA', 'linear_solver_MDA_options',
                       'linear_solver_MDO_options', 'group_mda_disciplines',
                       'transport_cost', 'transport_margin', 'year_start', 'year_end',
                       'energy_prices', 'energy_CO2_emissions', 'CO2_taxes', 'ressources_price',
                       'ressources_CO2_emissions', 'scaling_factor_techno_consumption',
                       'scaling_factor_techno_production', 'is_apply_ratio',
                       'is_stream_demand', 'is_apply_resource_ratio',
                       'residuals_history', 'all_streams_demand_ratio', 'all_resource_ratio_usable_demand']:
                inputs_dict[f'{namespace}.{key}'] = mda_data_input_dict[self.techno_name][key]['value']
                if mda_data_input_dict[self.techno_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{key}']
            else:
                inputs_dict[f'{namespace}.{self.techno_name}.{key}'] = mda_data_input_dict[self.techno_name][key]['value']
                if mda_data_input_dict[self.techno_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{self.techno_name}.{key}']

        pkl_file = open(
            join(dirname(__file__), 'data/mda_coarse_data_technologies_output_dict.pkl'), 'rb')
        mda_data_output_dict = pickle.load(pkl_file)
        pkl_file.close()

        coupled_outputs = []
        for key in mda_data_output_dict[self.techno_name].keys():
            # Modify namespace of output 'key' if needed
            if key in []:
                if mda_data_output_dict[self.techno_name][key]['is_coupling']:
                    coupled_outputs += [f'{namespace}.{key}']
            else:
                if mda_data_output_dict[self.techno_name][key]['is_coupling']:
                    coupled_outputs += [f'{namespace}.{self.techno_name}.{key}']

        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.techno_name}')[0]
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_{self.techno_name}.pkl',
                            discipline=disc, step=1.0e-18, derr_approx='complex_step', threshold=1e-5,
                            inputs=coupled_inputs,
                            outputs=coupled_outputs,)

    def test_02_coarse_fossil_techno_discipline_jacobian(self):
        '''
        Test the gradients of coarse fossil techno
        '''
        self.techno_name = 'FossilSimpleTechno'
        self.ee = ExecutionEngine(self.name)
        ns_dict = {'ns_public': self.name, 'ns_energy': self.name,
                   'ns_energy_study': f'{self.name}',
                   'ns_fossil': f'{self.name}',
                   'ns_resource': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'energy_models.models.fossil.fossil_simple_techno.fossil_simple_techno_disc.FossilSimpleTechnoDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.techno_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        pkl_file = open(
            join(dirname(__file__), 'data/mda_coarse_data_technologies_input_dict.pkl'), 'rb')
        mda_data_input_dict = pickle.load(pkl_file)
        pkl_file.close()

        namespace = f'{self.name}'
        inputs_dict = {}
        coupled_inputs = []
        for key in mda_data_input_dict[self.techno_name].keys():
            # Modify namespace of input 'key' if needed
            if key in ['linearization_mode', 'cache_type', 'cache_file_path', 'sub_mda_class',
                       'max_mda_iter', 'n_processes', 'chain_linearize', 'tolerance', 'use_lu_fact',
                       'warm_start', 'acceleration', 'warm_start_threshold', 'n_subcouplings_parallel',
                       'max_mda_iter_gs', 'condition_func', 'relax_factor', 'epsilon0', 'reset_history_each_run',
                       'linear_solver_MDO', 'linear_solver_MDA', 'linear_solver_MDA_options',
                       'linear_solver_MDO_options', 'group_mda_disciplines',
                       'transport_cost', 'transport_margin', 'year_start', 'year_end',
                       'energy_prices', 'energy_CO2_emissions', 'CO2_taxes', 'ressources_price',
                       'ressources_CO2_emissions', 'scaling_factor_techno_consumption',
                       'scaling_factor_techno_production', 'is_apply_ratio',
                       'is_stream_demand', 'is_apply_resource_ratio',
                       'residuals_history', 'all_streams_demand_ratio', 'all_resource_ratio_usable_demand']:
                inputs_dict[f'{namespace}.{key}'] = mda_data_input_dict[self.techno_name][key]['value']
                if mda_data_input_dict[self.techno_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{key}']
            else:
                inputs_dict[f'{namespace}.{self.techno_name}.{key}'] = mda_data_input_dict[self.techno_name][key]['value']
                if mda_data_input_dict[self.techno_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{self.techno_name}.{key}']

        pkl_file = open(
            join(dirname(__file__), 'data/mda_coarse_data_technologies_output_dict.pkl'), 'rb')
        mda_data_output_dict = pickle.load(pkl_file)
        pkl_file.close()

        coupled_outputs = []
        for key in mda_data_output_dict[self.techno_name].keys():
            # Modify namespace of output 'key' if needed
            if key in []:
                if mda_data_output_dict[self.techno_name][key]['is_coupling']:
                    coupled_outputs += [f'{namespace}.{key}']
            else:
                if mda_data_output_dict[self.techno_name][key]['is_coupling']:
                    coupled_outputs += [f'{namespace}.{self.techno_name}.{key}']

        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.techno_name}')[0]
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_{self.techno_name}.pkl',
                            discipline=disc, step=1.0e-18, derr_approx='complex_step', threshold=1e-5,
                            inputs=coupled_inputs,
                            outputs=coupled_outputs,)

    def test_03_coarse_dac_techno_discipline_jacobian(self):
        '''
        Test the gradients of coarse dac techno
        '''
        self.techno_name = 'direct_air_capture.DirectAirCaptureTechno'
        self.ee = ExecutionEngine(self.name)
        ns_dict = {'ns_public': self.name, 'ns_energy': self.name,
                   'ns_energy_study': f'{self.name}',
                   'ns_carbon_capture': f'{self.name}',
                   'ns_resource': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'energy_models.models.carbon_capture.direct_air_capture.direct_air_capture_techno.direct_air_capture_techno_disc.DirectAirCaptureTechnoDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.techno_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        pkl_file = open(
            join(dirname(__file__), 'data/mda_coarse_data_technologies_input_dict.pkl'), 'rb')
        mda_data_input_dict = pickle.load(pkl_file)
        pkl_file.close()

        namespace = f'{self.name}'
        inputs_dict = {}
        coupled_inputs = []
        for key in mda_data_input_dict[self.techno_name].keys():
            # Modify namespace of input 'key' if needed
            if key in ['linearization_mode', 'cache_type', 'cache_file_path', 'sub_mda_class',
                       'max_mda_iter', 'n_processes', 'chain_linearize', 'tolerance', 'use_lu_fact',
                       'warm_start', 'acceleration', 'warm_start_threshold', 'n_subcouplings_parallel',
                       'max_mda_iter_gs', 'condition_func', 'relax_factor', 'epsilon0', 'reset_history_each_run',
                       'linear_solver_MDO', 'linear_solver_MDA', 'linear_solver_MDA_options',
                       'linear_solver_MDO_options', 'group_mda_disciplines',
                       'transport_cost', 'transport_margin', 'year_start', 'year_end',
                       'energy_prices', 'energy_CO2_emissions', 'CO2_taxes', 'ressources_price',
                       'ressources_CO2_emissions', 'scaling_factor_techno_consumption',
                       'scaling_factor_techno_production', 'is_apply_ratio',
                       'is_stream_demand', 'is_apply_resource_ratio',
                       'residuals_history', 'all_streams_demand_ratio', 'all_resource_ratio_usable_demand']:
                inputs_dict[f'{namespace}.{key}'] = mda_data_input_dict[self.techno_name][key]['value']
                if mda_data_input_dict[self.techno_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{key}']
            else:
                inputs_dict[f'{namespace}.{self.techno_name}.{key}'] = mda_data_input_dict[self.techno_name][key]['value']
                if mda_data_input_dict[self.techno_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{self.techno_name}.{key}']

        pkl_file = open(
            join(dirname(__file__), 'data/mda_coarse_data_technologies_output_dict.pkl'), 'rb')
        mda_data_output_dict = pickle.load(pkl_file)
        pkl_file.close()

        coupled_outputs = []
        for key in mda_data_output_dict[self.techno_name].keys():
            # Modify namespace of output 'key' if needed
            if key in []:
                if mda_data_output_dict[self.techno_name][key]['is_coupling']:
                    coupled_outputs += [f'{namespace}.{key}']
            else:
                if mda_data_output_dict[self.techno_name][key]['is_coupling']:
                    coupled_outputs += [f'{namespace}.{self.techno_name}.{key}']

        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.techno_name}')[0]
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_{self.techno_name}.pkl',
                            discipline=disc, step=1.0e-18, derr_approx='complex_step', threshold=1e-5,
                            inputs=coupled_inputs,
                            outputs=coupled_outputs,)

    def test_04_coarse_flue_gas_capture_techno_discipline_jacobian(self):
        '''
        Test the gradients of coarse flue_gas_capture techno
        '''
        self.techno_name = 'flue_gas_capture.FlueGasTechno'
        self.ee = ExecutionEngine(self.name)
        ns_dict = {'ns_public': self.name, 'ns_energy': self.name,
                   'ns_energy_study': f'{self.name}',
                   'ns_flue_gas': f'{self.name}',
                   'ns_carbon_capture': f'{self.name}',
                   'ns_resource': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'energy_models.models.carbon_capture.flue_gas_capture.flue_gas_techno.flue_gas_techno_disc.FlueGasTechnoDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.techno_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        pkl_file = open(
            join(dirname(__file__), 'data/mda_coarse_data_technologies_input_dict.pkl'), 'rb')
        mda_data_input_dict = pickle.load(pkl_file)
        pkl_file.close()

        namespace = f'{self.name}'
        inputs_dict = {}
        coupled_inputs = []
        for key in mda_data_input_dict[self.techno_name].keys():
            # Modify namespace of input 'key' if needed
            if key in ['linearization_mode', 'cache_type', 'cache_file_path', 'sub_mda_class',
                       'max_mda_iter', 'n_processes', 'chain_linearize', 'tolerance', 'use_lu_fact',
                       'warm_start', 'acceleration', 'warm_start_threshold', 'n_subcouplings_parallel',
                       'max_mda_iter_gs', 'condition_func', 'relax_factor', 'epsilon0', 'reset_history_each_run',
                       'linear_solver_MDO', 'linear_solver_MDA', 'linear_solver_MDA_options',
                       'linear_solver_MDO_options', 'group_mda_disciplines',
                       'transport_cost', 'transport_margin', 'year_start', 'year_end',
                       'energy_prices', 'energy_CO2_emissions', 'CO2_taxes', 'ressources_price',
                       'ressources_CO2_emissions', 'scaling_factor_techno_consumption',
                       'scaling_factor_techno_production', 'is_apply_ratio',
                       'is_stream_demand', 'is_apply_resource_ratio',
                       'residuals_history', 'all_streams_demand_ratio', 'all_resource_ratio_usable_demand',
                       'flue_gas_mean']:
                inputs_dict[f'{namespace}.{key}'] = mda_data_input_dict[self.techno_name][key]['value']
                if mda_data_input_dict[self.techno_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{key}']
            else:
                inputs_dict[f'{namespace}.{self.techno_name}.{key}'] = mda_data_input_dict[self.techno_name][key]['value']
                if mda_data_input_dict[self.techno_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{self.techno_name}.{key}']

        pkl_file = open(
            join(dirname(__file__), 'data/mda_coarse_data_technologies_output_dict.pkl'), 'rb')
        mda_data_output_dict = pickle.load(pkl_file)
        pkl_file.close()

        coupled_outputs = []
        for key in mda_data_output_dict[self.techno_name].keys():
            # Modify namespace of output 'key' if needed
            if key in []:
                if mda_data_output_dict[self.techno_name][key]['is_coupling']:
                    coupled_outputs += [f'{namespace}.{key}']
            else:
                if mda_data_output_dict[self.techno_name][key]['is_coupling']:
                    coupled_outputs += [f'{namespace}.{self.techno_name}.{key}']

        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.techno_name}')[0]
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_{self.techno_name}.pkl',
                            discipline=disc, step=1.0e-18, derr_approx='complex_step', threshold=1e-5,
                            inputs=coupled_inputs,
                            outputs=coupled_outputs,)

    def test_05_coarse_carbon_storage_techno_discipline_jacobian(self):
        '''
        Test the gradients of coarse carbon_storage techno
        '''
        self.techno_name = 'CarbonStorageTechno'
        self.ee = ExecutionEngine(self.name)
        ns_dict = {'ns_public': self.name, 'ns_energy': self.name,
                   'ns_energy_study': f'{self.name}',
                   'ns_carbon_storage': f'{self.name}',
                   'ns_resource': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'energy_models.models.carbon_storage.carbon_storage_techno.carbon_storage_techno_disc.CarbonStorageTechnoDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.techno_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        pkl_file = open(
            join(dirname(__file__), 'data/mda_coarse_data_technologies_input_dict.pkl'), 'rb')
        mda_data_input_dict = pickle.load(pkl_file)
        pkl_file.close()

        namespace = f'{self.name}'
        inputs_dict = {}
        coupled_inputs = []
        for key in mda_data_input_dict[self.techno_name].keys():
            # Modify namespace of input 'key' if needed
            if key in ['linearization_mode', 'cache_type', 'cache_file_path', 'sub_mda_class',
                       'max_mda_iter', 'n_processes', 'chain_linearize', 'tolerance', 'use_lu_fact',
                       'warm_start', 'acceleration', 'warm_start_threshold', 'n_subcouplings_parallel',
                       'max_mda_iter_gs', 'condition_func', 'relax_factor', 'epsilon0', 'reset_history_each_run',
                       'linear_solver_MDO', 'linear_solver_MDA', 'linear_solver_MDA_options',
                       'linear_solver_MDO_options', 'group_mda_disciplines',
                       'transport_cost', 'transport_margin', 'year_start', 'year_end',
                       'energy_prices', 'energy_CO2_emissions', 'CO2_taxes', 'ressources_price',
                       'ressources_CO2_emissions', 'scaling_factor_techno_consumption',
                       'scaling_factor_techno_production', 'is_apply_ratio',
                       'is_stream_demand', 'is_apply_resource_ratio',
                       'residuals_history', 'all_streams_demand_ratio', 'all_resource_ratio_usable_demand', ]:
                inputs_dict[f'{namespace}.{key}'] = mda_data_input_dict[self.techno_name][key]['value']
                if mda_data_input_dict[self.techno_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{key}']
            else:
                inputs_dict[f'{namespace}.{self.techno_name}.{key}'] = mda_data_input_dict[self.techno_name][key]['value']
                if mda_data_input_dict[self.techno_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{self.techno_name}.{key}']

        pkl_file = open(
            join(dirname(__file__), 'data/mda_coarse_data_technologies_output_dict.pkl'), 'rb')
        mda_data_output_dict = pickle.load(pkl_file)
        pkl_file.close()

        coupled_outputs = []
        for key in mda_data_output_dict[self.techno_name].keys():
            # Modify namespace of output 'key' if needed
            if key in []:
                if mda_data_output_dict[self.techno_name][key]['is_coupling']:
                    coupled_outputs += [f'{namespace}.{key}']
            else:
                if mda_data_output_dict[self.techno_name][key]['is_coupling']:
                    coupled_outputs += [f'{namespace}.{self.techno_name}.{key}']

        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.techno_name}')[0]
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_{self.techno_name}.pkl',
                            discipline=disc, step=1.0e-18, derr_approx='complex_step', threshold=1e-5,
                            inputs=coupled_inputs,
                            outputs=coupled_outputs,)

    def test_06_coarse_renewable_stream_discipline_jacobian(self):
        '''
        Test the gradients of the coarse renewable stream
        '''
        self.energy_name = 'renewable'
        self.ee = ExecutionEngine(self.name)
        ns_dict = {'ns_public': f'{self.name}',
                   'ns_energy': f'{self.name}',
                   'ns_renewable': f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_resource': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'energy_models.core.stream_type.energy_disciplines.renewable_disc.RenewableDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.energy_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        pkl_file = open(
            join(dirname(__file__), 'data/mda_coarse_data_streams_input_dict.pkl'), 'rb')
        mda_data_input_dict = pickle.load(pkl_file)
        pkl_file.close()

        namespace = f'{self.name}'
        inputs_dict = {}
        coupled_inputs = []
        for key in mda_data_input_dict[self.energy_name].keys():
            if key in ['technologies_list', 'CO2_taxes', 'year_start', 'year_end',
                       'scaling_factor_energy_production', 'scaling_factor_energy_consumption',
                       'scaling_factor_techno_consumption', 'scaling_factor_techno_production', ]:
                inputs_dict[f'{namespace}.{key}'] = mda_data_input_dict[self.energy_name][key]['value']
                if mda_data_input_dict[self.energy_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{key}']
            else:
                inputs_dict[f'{namespace}.{self.energy_name}.{key}'] = mda_data_input_dict[self.energy_name][key]['value']
                if mda_data_input_dict[self.energy_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{self.energy_name}.{key}']

        pkl_file = open(
            join(dirname(__file__), 'data/mda_coarse_data_streams_output_dict.pkl'), 'rb')
        mda_data_output_dict = pickle.load(pkl_file)
        pkl_file.close()

        coupled_outputs = []
        for key in mda_data_output_dict[self.energy_name].keys():
            if mda_data_output_dict[self.energy_name][key]['is_coupling']:
                coupled_outputs += [f'{namespace}.{self.energy_name}.{key}']

        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.energy_name}')[0]
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_{self.energy_name}.pkl',
                            discipline=disc, step=1.0e-18, derr_approx='complex_step', threshold=1e-5,
                            inputs=coupled_inputs,
                            outputs=coupled_outputs,)

    def test_07_coarse_fossil_stream_discipline_jacobian(self):
        '''
        Test the gradients of the coarse fossil stream
        '''
        self.energy_name = 'fossil'
        self.ee = ExecutionEngine(self.name)
        ns_dict = {'ns_public': f'{self.name}',
                   'ns_fossil': f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   'ns_resource': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'energy_models.core.stream_type.energy_disciplines.fossil_disc.FossilDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.energy_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        pkl_file = open(
            join(dirname(__file__), 'data/mda_coarse_data_streams_input_dict.pkl'), 'rb')
        mda_data_input_dict = pickle.load(pkl_file)
        pkl_file.close()

        namespace = f'{self.name}'
        inputs_dict = {}
        coupled_inputs = []
        for key in mda_data_input_dict[self.energy_name].keys():
            if key in ['technologies_list', 'CO2_taxes', 'year_start', 'year_end',
                       'scaling_factor_energy_production', 'scaling_factor_energy_consumption',
                       'scaling_factor_techno_consumption', 'scaling_factor_techno_production', ]:
                inputs_dict[f'{namespace}.{key}'] = mda_data_input_dict[self.energy_name][key]['value']
                if mda_data_input_dict[self.energy_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{key}']
            else:
                inputs_dict[f'{namespace}.{self.energy_name}.{key}'] = mda_data_input_dict[self.energy_name][key]['value']
                if mda_data_input_dict[self.energy_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{self.energy_name}.{key}']

        pkl_file = open(
            join(dirname(__file__), 'data/mda_coarse_data_streams_output_dict.pkl'), 'rb')
        mda_data_output_dict = pickle.load(pkl_file)
        pkl_file.close()

        coupled_outputs = []
        for key in mda_data_output_dict[self.energy_name].keys():
            if mda_data_output_dict[self.energy_name][key]['is_coupling']:
                coupled_outputs += [f'{namespace}.{self.energy_name}.{key}']

        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.energy_name}')[0]
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_{self.energy_name}.pkl',
                            discipline=disc, step=1.0e-18, derr_approx='complex_step', threshold=1e-5,
                            inputs=coupled_inputs,
                            outputs=coupled_outputs,)


if '__main__' == __name__:
    AbstractJacobianUnittest.DUMP_JACOBIAN = True
    cls = CoarseJacobianTestCase()
    cls.setUp()
    cls.test_07_coarse_fossil_stream_discipline_jacobian()
