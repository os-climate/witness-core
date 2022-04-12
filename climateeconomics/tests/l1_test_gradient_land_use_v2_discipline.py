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
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest
import pandas as pd
import numpy as np


class LandUseV2JacobianDiscTest(AbstractJacobianUnittest):
    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_land_use_v2_discipline_analytic_grad
        ]

    def test_land_use_v2_discipline_analytic_grad(self):

        self.model_name = 'land_use_v2'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_land_use': f'{self.name}',
                   'ns_ref': f'{self.name}'
                   }

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_land_use.land_use.land_use_v2_disc.LandUseV2Discipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        land_demand_df = read_csv(
            join(data_dir, 'land_demandV2.csv'))

        year_start = 2020
        year_end = 2050
        years = np.arange(year_start, year_end + 1, 1)
        year_range = year_end - year_start + 1

        land_demand_df = land_demand_df.loc[land_demand_df['years']
                                            >= year_start]
        land_demand_df = land_demand_df.loc[land_demand_df['years']
                                            <= year_end]

        self.total_food_land_surface = pd.DataFrame(
            index=years,
            columns=['years',
                     'total surface (Gha)'])
        self.total_food_land_surface['years'] = years
        self.total_food_land_surface['total surface (Gha)'] = np.linspace(
            5, 4, year_range)

        initial_unsused_forest_surface = (4 - 1.25)
        self.forest_surface_df = pd.DataFrame(
            index=years,
            columns=['years',
                     'forest_constraint_evolution'])

        self.forest_surface_df['years'] = years
        # Gha
        self.forest_surface_df['forest_constraint_evolution'] = np.linspace(-0.5, 0, year_range)

        values_dict = {f'{self.name}.year_start': year_start,
                       f'{self.name}.year_end': year_end,
                       f'{self.name}.land_demand_df': land_demand_df,
                       f'{self.name}.total_food_land_surface': self.total_food_land_surface,
                       f'{self.name}.forest_surface_df': self.forest_surface_df,
                       f'{self.name}.initial_unsused_forest_surface': initial_unsused_forest_surface
                       }
        self.ee.dm.set_values_from_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.sos_disciplines[0]

        #AbstractJacobianUnittest.DUMP_JACOBIAN = True
        #np.set_printoptions(threshold=np.inf)
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_land_use_v2_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            inputs=[f'{self.name}.land_demand_df',
                                    f'{self.name}.forest_surface_df',
                                    f'{self.name}.total_food_land_surface',
                                    ],
                            outputs=[f'{self.name}.land_demand_constraint_df',
                                     f'{self.name}.land_surface_df',
                                     f'{self.name}.land_surface_for_food_df',
                                    ])
