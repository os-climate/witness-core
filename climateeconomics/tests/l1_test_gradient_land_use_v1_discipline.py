'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2023/11/03 Copyright 2023 Capgemini

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

from os.path import dirname, join

import numpy as np
import pandas as pd
from pandas import read_csv
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore


class LandUseV1JacobianDiscTest(AbstractJacobianUnittest):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_land_use_v1_discipline_analytic_grad
        ]

    def test_land_use_v1_discipline_analytic_grad(self):

        self.model_name = 'land_use_v1'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_land_use': f'{self.name}',}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_land_use.land_use.land_use_v1_disc.LandUseV1Discipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        land_demand_df = read_csv(
            join(data_dir, 'land_demand.csv'))

        year_start = GlossaryCore.YearStartDefault
        year_end = 2050
        years = np.arange(year_start, year_end + 1, 1)
        year_range = year_end - year_start + 1

        land_demand_df = land_demand_df.loc[land_demand_df[GlossaryCore.Years]
                                            >= year_start]
        land_demand_df = land_demand_df.loc[land_demand_df[GlossaryCore.Years]
                                            <= year_end]

        self.total_food_land_surface = pd.DataFrame(
            index=years,
            columns=[GlossaryCore.Years,
                     'total surface (Gha)'])
        self.total_food_land_surface[GlossaryCore.Years] = years
        self.total_food_land_surface['total surface (Gha)'] = np.linspace(
            5, 4, year_range)

        self.deforested_surface_df = pd.DataFrame(
            index=years,
            columns=[GlossaryCore.Years,
                     'forest_surface_evol'])
        self.deforested_surface_df[GlossaryCore.Years] = years
        # Gha
        self.deforested_surface_df['forest_surface_evol'] = np.linspace(
            -0.01, 0, year_range)

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': year_end,
                       f'{self.name}.land_demand_df': land_demand_df,
                       f'{self.name}.total_food_land_surface': self.total_food_land_surface,
                       f'{self.name}.forest_surface_df': self.deforested_surface_df
                       }
        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_land_use_v1_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data = disc_techno.local_data,
                            inputs=[f'{self.name}.land_demand_df',
                                    f'{self.name}.total_food_land_surface',
                                    f'{self.name}.forest_surface_df'],
                            outputs=[f'{self.name}.land_demand_constraint',
                                     f'{self.name}.land_surface_df',
                                     f'{self.name}.land_surface_for_food_df'])
