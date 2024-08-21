'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/14-2023/11/03 Copyright 2023 Capgemini

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
import pickle
from os.path import dirname, join

import numpy as np
import pandas as pd
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.tests.data.mda_coarse_data_generator import (
    launch_data_pickle_generation,
)


class CoarseJacobianTestCase(AbstractJacobianUnittest):
    """
    Ratio jacobian test class
    """

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

    def launch_data_pickle_generation(self):
        '''
        If the witness_coarse usecase changed, launch this function to update the data pickle
        '''
        launch_data_pickle_generation()

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'Test_Coarse'
        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1)
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
                       'max_mda_iter_gs', 'relax_factor', 'epsilon0',
                       'linear_solver_MDO', 'linear_solver_MDO_preconditioner', 'linear_solver_MDA',
                       'linear_solver_MDA_preconditioner', 'linear_solver_MDA_options',
                       'linear_solver_MDO_options', 'group_mda_disciplines',
                       'transport_cost', 'transport_margin', GlossaryCore.YearStart, GlossaryCore.YearEnd,
                       'energy_prices', GlossaryEnergy.StreamsCO2EmissionsValue, GlossaryCore.CO2TaxesValue, GlossaryCore.ResourcesPriceValue,
                       'resources_CO2_emissions', 'scaling_factor_techno_consumption',
                       'scaling_factor_techno_production', 'is_apply_ratio',
                       'is_stream_demand', 'is_apply_resource_ratio',
                       'residuals_history', 'all_streams_demand_ratio', 'all_resource_ratio_usable_demand']:
                inputs_dict[f'{namespace}.{key}'] = mda_data_input_dict[self.techno_name][key]['value']
                if mda_data_input_dict[self.techno_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{key}']
            else:
                inputs_dict[f'{namespace}.{self.techno_name}.{key}'] = mda_data_input_dict[self.techno_name][key][
                    'value']
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
            f'{self.name}.{self.techno_name}')[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_{self.techno_name}.pkl',
                            discipline=disc, step=1.0e-18, derr_approx='complex_step', threshold=1e-5,
                            local_data=disc.local_data,
                            inputs=coupled_inputs,
                            outputs=coupled_outputs, )

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
                       'max_mda_iter_gs', 'relax_factor', 'epsilon0',
                       'linear_solver_MDO', 'linear_solver_MDO_preconditioner', 'linear_solver_MDA',
                       'linear_solver_MDA_preconditioner', 'linear_solver_MDA_options',
                       'linear_solver_MDO_options', 'group_mda_disciplines',
                       'transport_cost', 'transport_margin', GlossaryCore.YearStart, GlossaryCore.YearEnd,
                       'energy_prices', GlossaryEnergy.StreamsCO2EmissionsValue, GlossaryCore.CO2TaxesValue, GlossaryCore.ResourcesPriceValue,
                       'resources_CO2_emissions', 'scaling_factor_techno_consumption',
                       'scaling_factor_techno_production', 'is_apply_ratio',
                       'is_stream_demand', 'is_apply_resource_ratio',
                       'residuals_history', 'all_streams_demand_ratio', 'all_resource_ratio_usable_demand']:
                inputs_dict[f'{namespace}.{key}'] = mda_data_input_dict[self.techno_name][key]['value']
                if mda_data_input_dict[self.techno_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{key}']
            else:
                inputs_dict[f'{namespace}.{self.techno_name}.{key}'] = mda_data_input_dict[self.techno_name][key][
                    'value']
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
            f'{self.name}.{self.techno_name}')[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_{self.techno_name}.pkl',
                            discipline=disc, step=1.0e-18, derr_approx='complex_step', threshold=1e-5,
                            local_data=disc.local_data,
                            inputs=coupled_inputs,
                            outputs=coupled_outputs, )

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
                       'max_mda_iter_gs', 'relax_factor', 'epsilon0',
                       'linear_solver_MDO', 'linear_solver_MDO_preconditioner', 'linear_solver_MDA',
                       'linear_solver_MDA_preconditioner', 'linear_solver_MDA_options',
                       'linear_solver_MDO_options', 'group_mda_disciplines',
                       'transport_cost', 'transport_margin', GlossaryCore.YearStart, GlossaryCore.YearEnd,
                       'energy_prices', GlossaryEnergy.StreamsCO2EmissionsValue, GlossaryCore.CO2TaxesValue, GlossaryCore.ResourcesPriceValue,
                       'resources_CO2_emissions', 'scaling_factor_techno_consumption',
                       'scaling_factor_techno_production', 'is_apply_ratio',
                       'is_stream_demand', 'is_apply_resource_ratio',
                       'residuals_history', 'all_streams_demand_ratio', 'all_resource_ratio_usable_demand']:
                inputs_dict[f'{namespace}.{key}'] = mda_data_input_dict[self.techno_name][key]['value']
                if mda_data_input_dict[self.techno_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{key}']
            else:
                inputs_dict[f'{namespace}.{self.techno_name}.{key}'] = mda_data_input_dict[self.techno_name][key][
                    'value']
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
            f'{self.name}.{self.techno_name}')[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_{self.techno_name}.pkl',
                            discipline=disc, step=1.0e-18, derr_approx='complex_step', threshold=1e-5,
                            local_data=disc.local_data,
                            inputs=coupled_inputs,
                            outputs=coupled_outputs, )

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
                       'max_mda_iter_gs', 'relax_factor', 'epsilon0',
                       'linear_solver_MDO', 'linear_solver_MDO_preconditioner', 'linear_solver_MDA',
                       'linear_solver_MDA_preconditioner', 'linear_solver_MDA_options',
                       'linear_solver_MDO_options', 'group_mda_disciplines',
                       'transport_cost', 'transport_margin', GlossaryCore.YearStart, GlossaryCore.YearEnd,
                       'energy_prices', GlossaryEnergy.StreamsCO2EmissionsValue, GlossaryCore.CO2TaxesValue, GlossaryCore.ResourcesPriceValue,
                       'resources_CO2_emissions', 'scaling_factor_techno_consumption',
                       'scaling_factor_techno_production', 'is_apply_ratio',
                       'is_stream_demand', 'is_apply_resource_ratio',
                       'residuals_history', 'all_streams_demand_ratio', 'all_resource_ratio_usable_demand',
                       'flue_gas_mean']:
                inputs_dict[f'{namespace}.{key}'] = mda_data_input_dict[self.techno_name][key]['value']
                if mda_data_input_dict[self.techno_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{key}']
            else:
                inputs_dict[f'{namespace}.{self.techno_name}.{key}'] = mda_data_input_dict[self.techno_name][key][
                    'value']
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
            f'{self.name}.{self.techno_name}')[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_{self.techno_name}.pkl',
                            discipline=disc, step=1.0e-18, derr_approx='complex_step', threshold=1e-5,
                            local_data=disc.local_data,
                            inputs=coupled_inputs,
                            outputs=coupled_outputs, )

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
                       'max_mda_iter_gs', 'relax_factor', 'epsilon0',
                       'linear_solver_MDO', 'linear_solver_MDO_preconditioner', 'linear_solver_MDA',
                       'linear_solver_MDA_preconditioner', 'linear_solver_MDA_options',
                       'linear_solver_MDO_options', 'group_mda_disciplines',
                       'transport_cost', 'transport_margin', GlossaryCore.YearStart, GlossaryCore.YearEnd,
                       'energy_prices', GlossaryEnergy.StreamsCO2EmissionsValue, GlossaryCore.CO2TaxesValue, GlossaryCore.ResourcesPriceValue,
                       'resources_CO2_emissions', 'scaling_factor_techno_consumption',
                       'scaling_factor_techno_production', 'is_apply_ratio',
                       'is_stream_demand', 'is_apply_resource_ratio',
                       'residuals_history', 'all_streams_demand_ratio', 'all_resource_ratio_usable_demand', ]:
                inputs_dict[f'{namespace}.{key}'] = mda_data_input_dict[self.techno_name][key]['value']
                if mda_data_input_dict[self.techno_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{key}']
            else:
                inputs_dict[f'{namespace}.{self.techno_name}.{key}'] = mda_data_input_dict[self.techno_name][key][
                    'value']
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
            f'{self.name}.{self.techno_name}')[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_{self.techno_name}.pkl',
                            discipline=disc, step=1.0e-18, derr_approx='complex_step', threshold=1e-5,
                            local_data=disc.local_data,
                            inputs=coupled_inputs,
                            outputs=coupled_outputs, )

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
            if key in [GlossaryCore.techno_list, GlossaryCore.CO2TaxesValue, GlossaryCore.YearStart,
                       GlossaryCore.YearEnd,
                       'scaling_factor_energy_production', 'scaling_factor_energy_consumption',
                       'scaling_factor_techno_consumption', 'scaling_factor_techno_production', ]:
                inputs_dict[f'{namespace}.{key}'] = mda_data_input_dict[self.energy_name][key]['value']
                if mda_data_input_dict[self.energy_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{key}']
            else:
                inputs_dict[f'{namespace}.{self.energy_name}.{key}'] = mda_data_input_dict[self.energy_name][key][
                    'value']
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

        technos = inputs_dict[f"{self.name}.technologies_list"]
        techno_capital = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.Capital: 20000 * np.ones_like(self.years)
        })
        for techno in technos:
            inputs_dict[
                f"{self.name}.{self.energy_name}.{techno}.{GlossaryEnergy.TechnoCapitalValue}"] = techno_capital
            coupled_inputs.append(f"{self.name}.{self.energy_name}.{techno}.{GlossaryEnergy.TechnoCapitalValue}")

        coupled_outputs.append(f"{self.name}.{self.energy_name}.{GlossaryEnergy.EnergyTypeCapitalDfValue}")

        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.energy_name}')[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_{self.energy_name}.pkl',
                            discipline=disc, step=1.0e-18, derr_approx='complex_step', threshold=1e-5,
                            local_data=disc.local_data,
                            inputs=coupled_inputs,
                            outputs=coupled_outputs, )

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
            if key in [GlossaryCore.techno_list, GlossaryCore.CO2TaxesValue, GlossaryCore.YearStart,
                       GlossaryCore.YearEnd,
                       'scaling_factor_energy_production', 'scaling_factor_energy_consumption',
                       'scaling_factor_techno_consumption', 'scaling_factor_techno_production', ]:
                inputs_dict[f'{namespace}.{key}'] = mda_data_input_dict[self.energy_name][key]['value']
                if mda_data_input_dict[self.energy_name][key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{key}']
            else:
                inputs_dict[f'{namespace}.{self.energy_name}.{key}'] = mda_data_input_dict[self.energy_name][key][
                    'value']
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

        technos = inputs_dict[f"{self.name}.technologies_list"]
        techno_capital = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.Capital: 20000 * np.ones_like(self.years)
        })
        for techno in technos:
            inputs_dict[
                f"{self.name}.{self.energy_name}.{techno}.{GlossaryEnergy.TechnoCapitalValue}"] = techno_capital
            coupled_inputs.append(f"{self.name}.{self.energy_name}.{techno}.{GlossaryEnergy.TechnoCapitalValue}")

        coupled_outputs.append(f"{self.name}.{self.energy_name}.{GlossaryEnergy.EnergyTypeCapitalDfValue}")
        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.energy_name}')[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_{self.energy_name}.pkl',
                            discipline=disc, step=1.0e-18, derr_approx='complex_step', threshold=1e-5,
                            local_data=disc.local_data,
                            inputs=coupled_inputs,
                            outputs=coupled_outputs, )

    def test_08_coarse_energymix_discipline_jacobian(self):
        '''
        Test the gradients of the Energy mix discipline when coarse
        '''
        self.energy_name = 'energymix_coarse'
        self.ee = ExecutionEngine(self.name)
        ns_dict = {'ns_public': f'{self.name}',
                   'ns_hydrogen': f'{self.name}',
                   'ns_syngas': f'{self.name}',
                   'ns_methane': f'{self.name}',
                   GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_energy_study': f'{self.name}.{self.energy_name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}.{self.energy_name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}.{self.energy_name}',
                   'ns_resource': f'{self.name}.{self.energy_name}.resource',
                   GlossaryCore.NS_CCS: f'{self.name}.{self.energy_name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}.{self.energy_name}',
                   'ns_energy': f'{self.name}.{self.energy_name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'energy_models.core.energy_mix.energy_mix_disc.Energy_Mix_Discipline'
        builder = self.ee.factory.get_builder_from_module(
            self.energy_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        pkl_file = open(
            join(dirname(__file__), 'data/mda_coarse_data_energymix_input_dict.pkl'), 'rb')
        mda_data_input_dict = pickle.load(pkl_file)
        pkl_file.close()
        namespace = f'{self.name}'
        inputs_dict = {}
        coupled_inputs = []
        for key in mda_data_input_dict.keys():
            if key in [GlossaryCore.techno_list, GlossaryCore.YearStart, GlossaryCore.YearEnd,
                       'scaling_factor_energy_production', 'scaling_factor_energy_consumption',
                       'scaling_factor_techno_consumption', 'scaling_factor_techno_production', ]:
                inputs_dict[f'{namespace}.{key}'] = mda_data_input_dict[key]['value']
                if mda_data_input_dict[key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{key}']
            else:
                inputs_dict[f'{namespace}.{self.energy_name}.{key}'] = mda_data_input_dict[key]['value']
                if mda_data_input_dict[key]['is_coupling']:
                    coupled_inputs += [f'{namespace}.{self.energy_name}.{key}']

        pkl_file = open(
            join(dirname(__file__), 'data/mda_coarse_data_energymix_output_dict.pkl'), 'rb')
        mda_data_output_dict = pickle.load(pkl_file)
        pkl_file.close()
        excluded_elem = ['resources_demand', 'resources_demand_woratio']
        coupled_outputs = []
        for key in mda_data_output_dict.keys():
            if mda_data_output_dict[key]['is_coupling'] and key not in excluded_elem:
                coupled_outputs += [f'{namespace}.{self.energy_name}.{key}']

        energy_types = inputs_dict[f"{self.name}.{self.energy_name}.energy_list"]
        techno_capital = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.Capital: 20000 * np.ones_like(self.years)
        })
        for energy in energy_types:
            inputs_dict[
                f"{self.name}.{self.energy_name}.{energy}.{GlossaryEnergy.EnergyTypeCapitalDfValue}"] = techno_capital
            coupled_inputs.append(f"{self.name}.{self.energy_name}.{energy}.{GlossaryEnergy.EnergyTypeCapitalDfValue}")

        for energy in inputs_dict[f"{self.name}.{self.energy_name}.ccs_list"]:
            inputs_dict[
                f"{self.name}.{self.energy_name}.{energy}.{GlossaryEnergy.EnergyTypeCapitalDfValue}"] = techno_capital
            coupled_inputs.append(f"{self.name}.{self.energy_name}.{energy}.{GlossaryEnergy.EnergyTypeCapitalDfValue}")

        coupled_outputs.append(f"{self.name}.{GlossaryEnergy.EnergyCapitalDfValue}")
        coupled_outputs.remove(f"{self.name}.energymix_coarse.{GlossaryEnergy.EnergyCapitalDfValue}")
        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.energy_name}')[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_{self.energy_name}.pkl',
                            discipline=disc, step=1.0e-15, derr_approx='complex_step', threshold=1e-5,
                            local_data=disc.local_data,
                            inputs=coupled_inputs,
                            outputs=coupled_outputs, )

    def _test_problematic_optim_point(self):
        self.energy_name = 'energymix_coarse'
        self.ee = ExecutionEngine(self.name)
        ns_dict = {'ns_public': f'{self.name}',
                   'ns_hydrogen': f'{self.name}',
                   'ns_syngas': f'{self.name}',
                   'ns_methane': f'{self.name}',
                   GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   'ns_resource': f'{self.name}',
                   GlossaryCore.NS_CCS: f'{self.name}.{self.energy_name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   'ns_energy': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'energy_models.core.energy_mix.energy_mix_disc.Energy_Mix_Discipline'
        builder = self.ee.factory.get_builder_from_module(self.energy_name, mod_path)
        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        import os
        import pickle
        with open(os.path.join("data", "uc1optim.pkl"), "rb") as f:
            dict_input_optimized_point = pickle.load(f)

        def find_var_in_dict(varname: str):
            try:
                varname_in_dict_optimized = list(filter(lambda x: varname in x, dict_input_optimized_point.keys()))[0]
                var_value = dict_input_optimized_point[varname_in_dict_optimized]
                return var_value
            except IndexError:
                print(varname)

        input_list = [GlossaryEnergy.energy_list, GlossaryEnergy.ccs_list, GlossaryEnergy.CO2TaxesValue, GlossaryCore.YearStart, GlossaryCore.YearEnd]
        inputs_dict = {}

        checked_input_jacobian = []

        for stream in GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT.keys():
            to_have = [GlossaryEnergy.EnergyConsumptionValue,
                       GlossaryCore.EnergyConsumptionWithoutRatioValue,
                       GlossaryCore.EnergyProductionValue,
                       GlossaryCore.StreamPricesValue,
                       GlossaryCore.LandUseRequiredValue,
                       GlossaryEnergy.EnergyTypeCapitalDfValue,
                       GlossaryEnergy.CO2EmissionsValue,
                       GlossaryEnergy.CO2PerUse,
                       ]

            to_check_jacobian = [
                GlossaryCore.EnergyProductionValue,
                GlossaryCore.EnergyConsumptionValue,
            ]

            input_list.extend([f"{stream}.{_}" for _ in to_have])
            checked_input_jacobian.extend([f"{self.name}.{self.energy_name}.{stream}.{_}" for _ in to_check_jacobian])

        for checked_input in input_list:
            var_value = find_var_in_dict(checked_input)

            varname_in_input_dicts = f'{self.name}.{checked_input}'
            inputs_dict.update({varname_in_input_dicts: var_value})
            varname_in_input_dicts = f'{self.name}.{self.energy_name}.{checked_input}'
            inputs_dict.update({varname_in_input_dicts: var_value})


            a = 1

        inputs_dict.update({
            f'{self.name}.{GlossaryCore.YearEnd}': find_var_in_dict(GlossaryCore.YearEnd),
        })

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.energy_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            # graph.to_plotly().show()
            pass

        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_at_opt_point_uc1_energy_mix_disc.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=checked_input_jacobian,
                            outputs=[f"{self.name}.{GlossaryCore.EnergyMeanPriceValue}",
                                     f"{self.name}.{self.energy_name}.{GlossaryCore.EnergyProductionValue}"],)

    def _test_problematic_optim_point_2(self):
        self.energy_name = 'energymix_coarse'
        self.ee = ExecutionEngine(self.name)
        ns_dict = {'ns_public': f'{self.name}',
                   'ns_hydrogen': f'{self.name}',
                   'ns_syngas': f'{self.name}',
                   'ns_methane': f'{self.name}',
                   GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_energy_study': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   'ns_resource': f'{self.name}',
                   GlossaryCore.NS_CCS: f'{self.name}.{self.energy_name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   'ns_energy': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'energy_models.core.energy_mix.energy_mix_disc.Energy_Mix_Discipline'
        builder = self.ee.factory.get_builder_from_module(self.energy_name, mod_path)
        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        import os
        import pickle
        with open(os.path.join("data", "uc4optim.pkl"), "rb") as f:
            dict_input_optimized_point = pickle.load(f)

        def find_var_in_dict(varname: str):
            try:
                varname_in_dict_optimized = list(filter(lambda x: varname in x, dict_input_optimized_point.keys()))[0]
                var_value = dict_input_optimized_point[varname_in_dict_optimized]
                return var_value
            except IndexError:
                print(varname)

        input_list = [GlossaryEnergy.energy_list, GlossaryEnergy.ccs_list, GlossaryEnergy.CO2TaxesValue, GlossaryCore.YearStart, GlossaryCore.YearEnd]
        inputs_dict = {}

        checked_input_jacobian = []

        for stream in GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT.keys():
            to_have = [GlossaryEnergy.EnergyConsumptionValue,
                       GlossaryCore.EnergyConsumptionWithoutRatioValue,
                       GlossaryCore.EnergyProductionValue,
                       GlossaryCore.StreamPricesValue,
                       GlossaryCore.LandUseRequiredValue,
                       GlossaryEnergy.EnergyTypeCapitalDfValue,
                       GlossaryEnergy.CO2EmissionsValue,
                       GlossaryEnergy.CO2PerUse,
                       ]

            to_check_jacobian = [
                GlossaryCore.EnergyProductionValue,
                GlossaryCore.EnergyConsumptionValue,
            ]

            input_list.extend([f"{stream}.{_}" for _ in to_have])
            checked_input_jacobian.extend([f"{self.name}.{self.energy_name}.{stream}.{_}" for _ in to_check_jacobian])

        for checked_input in input_list:
            var_value = find_var_in_dict(checked_input)

            varname_in_input_dicts = f'{self.name}.{checked_input}'
            inputs_dict.update({varname_in_input_dicts: var_value})
            varname_in_input_dicts = f'{self.name}.{self.energy_name}.{checked_input}'
            inputs_dict.update({varname_in_input_dicts: var_value})


            a = 1

        inputs_dict.update({
            f'{self.name}.{GlossaryCore.YearEnd}': find_var_in_dict(GlossaryCore.YearEnd),
        })

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.energy_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            # graph.to_plotly().show()
            pass

        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_at_opt_point_uc4_energy_mix_disc.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            local_data=disc_techno.local_data,
                            inputs=checked_input_jacobian,
                            outputs=[f"{self.name}.{GlossaryCore.EnergyMeanPriceValue}",
                                     f"{self.name}.{self.energy_name}.{GlossaryCore.EnergyProductionValue}"],)


if '__main__' == __name__:
    cls = CoarseJacobianTestCase()
    cls.setUp()
    # self.launch_data_pickle_generation()
    cls.test_08_coarse_energymix_discipline_jacobian()
