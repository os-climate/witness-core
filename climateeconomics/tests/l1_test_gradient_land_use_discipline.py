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
from os.path import join, dirname
from pandas import read_csv
from pathlib import Path
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest
import unittest
import pandas as pd
import numpy as np


class LandUseJacobianDiscTest(AbstractJacobianUnittest):
    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_land_use_discipline_analytic_grad
        ]

    def test_land_use_discipline_analytic_grad(self):

        self.model_name = 'land_use'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_land_use': f'{self.name}',
                   'ns_ref': f'{self.name}'
                   }

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_land_use.land_use.land_use_disc.LandUseDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        land_demand_df = read_csv(
            join(data_dir, 'land_demand.csv'))

        year_start = 2020
        year_end = 2050
        years = np.arange(year_start, year_end + 1, 1)
        year_range = year_end - year_start + 1

        land_demand_df = land_demand_df.loc[land_demand_df['years']
                                            >= year_start]
        land_demand_df = land_demand_df.loc[land_demand_df['years']
                                            <= year_end]

        global_data_dir = join(Path(__file__).parents[1], 'data')
        population_df = pd.read_csv(
            join(global_data_dir, 'population_df.csv'))
        population_df = population_df.loc[population_df['years']
                                          >= year_start]
        population_df = population_df.loc[population_df['years']
                                          <= year_end]
        population_df.index = years

        percentage = np.linspace(70, 50, year_range)
        livestock_usage_factor_df = pd.DataFrame(
            {'percentage': percentage})
        livestock_usage_factor_df.index = years

        values_dict = {f'{self.name}.year_start': year_start,
                       f'{self.name}.year_end': year_end,
                       f'{self.name}.land_demand_df': land_demand_df,
                       f'{self.name}.population_df': population_df,
                       f'{self.name}.land_use.crop_land_use_per_capita': 0.21,
                       f'{self.name}.land_use.livestock_land_use_per_capita': 0.42,
                       f'{self.name}.livestock_usage_factor_df': livestock_usage_factor_df,
                       }
        self.ee.dm.set_values_from_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.sos_disciplines[0]

        #AbstractJacobianUnittest.DUMP_JACOBIAN = True
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_land_use_discipline.pkl', discipline=disc_techno, inputs=[f'{self.name}.land_demand_df', f'{self.name}.livestock_usage_factor_df'], outputs=[
                            f'{self.name}.land_demand_constraint_df', f'{self.name}.land_surface_df', f'{self.name}.land_surface_for_food_df'], step=1e-15, derr_approx='complex_step')
