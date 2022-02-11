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

import unittest
import numpy as np
import pandas as pd
from os.path import join, dirname
from pandas import DataFrame, read_csv

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class CarbonEmissionsJacobianDiscTest(AbstractJacobianUnittest):
    # AbstractJacobianUnittest.DUMP_JACOBIAN = True
    # np.set_printoptions(threshold=np.inf)

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_carbon_emissions_analytic_grad,
            self.test_co2_objective_limit_grad
        ]

    def test_carbon_emissions_analytic_grad(self):

        self.model_name = 'carbonemission'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_ref': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.carbonemissions.carbonemissions_discipline.CarbonemissionsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')

        economics_df_all = read_csv(
            join(data_dir, 'economics_data_onestep.csv'))
        energy_supply_df_all = read_csv(
            join(data_dir, 'energy_supply_data_onestep.csv'))
        year_start = 2020
        economics_df_y = economics_df_all[economics_df_all['years'] >= year_start][[
            'years', 'gross_output']]
        energy_supply_df_y = energy_supply_df_all[energy_supply_df_all['years'] >= year_start][[
            'years', 'total_CO2_emitted']]
        energy_supply_df_y["years"] = energy_supply_df_all['years']
        energy_supply_df_y = energy_supply_df_y.rename(
            columns={'total_CO2_emitted': 'Total CO2 emissions'})

        # put manually the index
        years = np.arange(year_start, 2101)
        economics_df_y.index = years
        energy_supply_df_y.index = years

        CO2_emitted_forest = pd.DataFrame()
        emission_forest = np.linspace(10, 100, len(years))
        cum_emission = np.cumsum(emission_forest) + 2850
        CO2_emitted_forest['years'] = years
        CO2_emitted_forest['emitted_CO2_evol_cumulative'] = cum_emission

        values_dict = {f'{self.name}.economics_df': economics_df_y,
                       f'{self.name}.co2_emissions_Gt': energy_supply_df_y,
                       f'{self.name}.CO2_emitted_forest_df': CO2_emitted_forest, }

        self.ee.load_study_from_input_dict(values_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_carbon_emission_discipline.pkl', 
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            inputs=[f'{self.name}.economics_df', 
                                    f'{self.name}.co2_emissions_Gt', 
                                    f'{self.name}.CO2_emitted_forest_df'],
                            outputs=[f'{self.name}.emissions_df', 
                                    f'{self.name}.CO2_objective'])

    def test_co2_objective_limit_grad(self):

        self.model_name = 'carbonemission'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_ref': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.carbonemissions.carbonemissions_discipline.CarbonemissionsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')

        economics_df_all = read_csv(
            join(data_dir, 'economics_data_onestep.csv'))
        energy_supply_df_all = read_csv(
            join(data_dir, 'energy_supply_data_onestep.csv'))

        economics_df_y = economics_df_all[economics_df_all['years'] >= 2020][[
            'years', 'gross_output']]
        energy_supply_df_y = energy_supply_df_all[energy_supply_df_all['years'] >= 2020][[
            'years', 'total_CO2_emitted']]
        energy_supply_df_y["years"] = energy_supply_df_all['years']
        energy_supply_df_y = energy_supply_df_y.rename(
            columns={'total_CO2_emitted': 'Total CO2 emissions'})

        # put manually the index
        years = np.arange(2020, 2101)
        economics_df_y.index = years
        energy_supply_df_y.index = years
        energy_supply_df_y['Total CO2 emissions'] = np.linspace(
            0, -3000, len(years))

        CO2_emitted_forest = pd.DataFrame()
        emission_forest = np.linspace(40, 40, len(years))
        cum_emission = np.cumsum(emission_forest) + 2850
        CO2_emitted_forest['years'] = years
        CO2_emitted_forest['emitted_CO2_evol_cumulative'] = cum_emission

        values_dict = {f'{self.name}.economics_df': economics_df_y,
                       f'{self.name}.co2_emissions_Gt': energy_supply_df_y,
                       f'{self.name}.CO2_emitted_forest_df': CO2_emitted_forest, }

        self.ee.load_study_from_input_dict(values_dict)
        disc_techno = self.ee.root_process.sos_disciplines[0]

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_co2_objective_limit.pkl', 
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', 
                            inputs=[f'{self.name}.economics_df', 
                                    f'{self.name}.co2_emissions_Gt', 
                                    f'{self.name}.CO2_emitted_forest_df'],
                            outputs=[f'{self.name}.emissions_df',
                                    f'{self.name}.CO2_objective'])
