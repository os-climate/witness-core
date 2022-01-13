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


class UtilityJacobianDiscTest(AbstractJacobianUnittest):
    # AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):
        self.name = 'Test'
        self.model_name = 'utility'
        self.year_start = 2020
        self.year_end = 2100
        self.years = np.arange(self.year_start, self.year_end + 1)
        self.year_range = self.year_end - self.year_start

        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_ref': f'{self.name}'}
        self.ee = ExecutionEngine(self.name)
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.utilitymodel.utilitymodel_discipline.UtilityModelDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        economics_df_all = read_csv(
            join(data_dir, 'economics_data_onestep.csv'))

        economics_df_y = economics_df_all[economics_df_all['years'] >= 2020]
        economics_df = economics_df_y[[
            'years', 'pc_consumption']]
        years = np.arange(2020, 2101, 1)
        economics_df.index = years

        global_data_dir = join(dirname(dirname(__file__)), 'data')
        self.population_df = read_csv(
            join(global_data_dir, 'population_df.csv'))
        self.population_df.index = years

        energy_price = np.linspace(200, 10, len(years))
        energy_mean_price = pd.DataFrame(
            {'years': years, 'energy_price': energy_price})

        self.values_dict = {f'{self.name}.economics_df': economics_df,
                            f'{self.name}.population_df': self.population_df,
                            f'{self.name}.energy_mean_price': energy_mean_price}

        self.ee.load_study_from_input_dict(self.values_dict)

        self.disc_techno = self.ee.root_process.sos_disciplines[0]

    def analytic_grad_entry(self):
        return [
            self.test_01_utility_analytic_grad_welfare,
            self.test_02_utility_analytic_grad_last_utility,
            self.test_03_utility_with_low_economy
        ]

    def test_01_utility_analytic_grad_welfare(self):

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_utility_discipline_welfare.pkl', discipline=self.disc_techno, step=1e-15,
                            inputs=[f'{self.name}.economics_df',
                                    f'{self.name}.energy_mean_price',
                                    f'{self.name}.population_df'],
                            outputs=[f'{self.name}.welfare_objective',
                                     f'{self.name}.min_utility_objective',
                                     f'{self.name}.utility_df'],
                            derr_approx='complex_step')

    def test_02_utility_analytic_grad_last_utility(self):
        """
        Test the second option of the objective function
        """

        self.values_dict[f'{self.name}.welfare_obj_option'] = 'last_utility'

        self.ee.load_study_from_input_dict(self.values_dict)

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_utility_discipline_last_utility.pkl', discipline=self.disc_techno, step=1e-15,
                            inputs=[f'{self.name}.economics_df',
                                    f'{self.name}.energy_mean_price',
                                    f'{self.name}.population_df'],
                            outputs=[f'{self.name}.welfare_objective',
                                     f'{self.name}.min_utility_objective',
                                     f'{self.name}.utility_df'],
                            derr_approx='complex_step')

    def test_03_utility_with_low_economy(self):

        data_dir = join(dirname(__file__), 'data')
        economics_df_all = read_csv(
            join(data_dir, 'economics_data_onestep.csv'))

        economics_df_y = economics_df_all[economics_df_all['years'] >= 2020]
        economics_df = economics_df_y[[
            'years', 'pc_consumption']]
        economics_df['pc_consumption'] = economics_df['pc_consumption'] / 2
        years = np.arange(2020, 2101, 1)
        economics_df.index = years
        energy_price = np.arange(200, 200 + len(years))
        energy_mean_price = pd.DataFrame(
            {'years': years, 'energy_price': energy_price})

        values_dict = {f'{self.name}.economics_df': economics_df,
                       f'{self.name}.population_df': self.population_df,
                       f'{self.name}.energy_mean_price': energy_mean_price}

        self.ee.load_study_from_input_dict(values_dict)

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_utility_low_economy.pkl', discipline=self.disc_techno, step=1e-15,
                            inputs=[f'{self.name}.economics_df',
                                    f'{self.name}.energy_mean_price',
                                    f'{self.name}.population_df'],
                            outputs=[f'{self.name}.welfare_objective',
                                     f'{self.name}.min_utility_objective',
                                     f'{self.name}.utility_df'],
                            derr_approx='complex_step')
