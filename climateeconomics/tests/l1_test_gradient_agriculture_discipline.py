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

from os.path import dirname

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore


class AgricultureJacobianDiscTest(AbstractJacobianUnittest):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_agriculture_discipline_analytic_grad
        ]

    def test_agriculture_discipline_analytic_grad(self):

        self.model_name = 'agriculture'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_agriculture': f'{self.name}'
                   }

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.agriculture.agriculture_disc.AgricultureDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.year_start =GlossaryCore.YearStartDefault
        self.year_end = 2055
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1

        population = np.array(np.linspace(8000, 9000, year_range))
        self.population_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.PopulationValue: population})
        self.population_df.index = years

        temperature = np.array(np.linspace(1.05, 5, year_range))
        self.temperature_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.TempAtmo: temperature})
        self.temperature_df.index = years

        self.default_kg_to_m2 = {'red meat': 360,
                                 'white meat': 16,
                                 'milk': 8.95,
                                 'eggs': 6.3,
                                 'rice and maize': 2.9,
                                 'potatoes': 0.88,
                                 'fruits and vegetables': 0.8,
                                 }
        self.default_kg_to_kcal = {'red meat': 2566,
                                   'white meat': 1860,
                                   'milk': 550,
                                   'eggs': 1500,
                                   'rice and maize': 1150,
                                   'potatoes': 670,
                                   'fruits and vegetables': 624,
                                   }
        red_meat_percentage = np.linspace(6, 1, year_range)
        white_meat_percentage = np.linspace(14, 5, year_range)
        self.red_meat_percentage = pd.DataFrame({
                            GlossaryCore.Years: years,
                            'red_meat_percentage': red_meat_percentage})
        self.white_meat_percentage = pd.DataFrame({
                                GlossaryCore.Years: years,
                                'white_meat_percentage': white_meat_percentage})

        self.other = np.linspace(0.08, 0.08, year_range)

        self.diet_df = pd.DataFrame({'red meat': [11.02],
                                     'white meat': [31.11],
                                     'milk': [79.27],
                                     'eggs': [9.68],
                                     'rice and maize': [97.76],
                                     'potatoes': [32.93],
                                     'fruits and vegetables': [217.62],
                                     })

        self.param = {GlossaryCore.YearStart: self.year_start,
                      GlossaryCore.YearEnd: self.year_end,
                      GlossaryCore.TimeStep: self.time_step,
                      'diet_df': self.diet_df,
                      'kg_to_kcal_dict': self.default_kg_to_kcal,
                      GlossaryCore.PopulationDfValue: self.population_df,
                      'kg_to_m2_dict': self.default_kg_to_m2,
                      'red_meat_percentage': self.red_meat_percentage,
                      'white_meat_percentage': self.white_meat_percentage,
                      'other_use_agriculture': self.other
                      }

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{self.model_name}.diet_df': self.diet_df,
                       f'{self.name}.{self.model_name}.kg_to_kcal_dict': self.default_kg_to_kcal,
                       f'{self.name}.{self.model_name}.kg_to_m2_dict': self.default_kg_to_m2,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                       f'{self.name}.{GlossaryCore.TemperatureDfValue}': self.temperature_df,
                       f'{self.name}.red_meat_percentage': self.red_meat_percentage,
                       f'{self.name}.white_meat_percentage': self.white_meat_percentage,
                       f'{self.name}.{self.model_name}.other_use_agriculture': self.other,
                       }
        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_agriculture_discipline.pkl', discipline=disc_techno,
                            step=1e-15, derr_approx='complex_step',local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.TemperatureDfValue}',
                                    f'{self.name}.red_meat_percentage',
                                    f'{self.name}.white_meat_percentage',
                                    ],
                            outputs=[f'{self.name}.total_food_land_surface'])
