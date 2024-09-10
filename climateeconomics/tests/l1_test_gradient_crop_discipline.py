'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/21-2023/11/03 Copyright 2023 Capgemini

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
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore


class AgricultureJacobianDiscTest(AbstractJacobianUnittest):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        '''
        Initialize third data needed for testing
        '''
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2035
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1

        population = np.array(np.linspace(7800, 7900, year_range))
        self.population_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.PopulationValue: population})
        self.population_df.index = years
        temperature = np.array(np.linspace(1.05, 5, year_range))
        self.temperature_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.TempAtmo: temperature})
        self.temperature_df.index = years

        # Age distribution of forests in 2008 (
        self.default_kg_to_m2 = {'red meat': 345.,
                            'white meat': 14.5,
                            'milk': 8.95,
                            'eggs': 6.27,
                            'rice and maize': 2.89,
                            'cereals': 4.5,
                            'fruits and vegetables': 0.8,
                            GlossaryCore.Fish: 0.,
                            GlossaryCore.OtherFood: 5.1041,
                            }
        # land use of other is provided in variable 'other_use_crop'

        self.default_kg_to_kcal = {'red meat': 1551.05,
                              'white meat': 2131.99,
                              'milk': 921.76,
                              'eggs': 1425.07,
                              'rice and maize': 2572.46,
                              'cereals': 2964.99,
                              'fruits and vegetables': 559.65,
                              GlossaryCore.Fish: 609.17,
                              GlossaryCore.OtherFood: 3061.06,
                              }

        self.diet_df = pd.DataFrame({"red meat": [11.02],
                                    "white meat": [31.11],
                                    "milk": [79.27],
                                    "eggs": [9.68],
                                    "rice and maize": [98.08],
                                    "cereals": [78],
                                    "fruits and vegetables": [293],
                                    GlossaryCore.Fish: [23.38],
                                    GlossaryCore.OtherFood: [77.24]
                                    })

        # investment: 1Mha of crop land each year
        self.crop_investment = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: np.ones(len(years)) * 0.381})

        self.margin = pd.DataFrame(
            {GlossaryCore.Years: years, 'margin': np.ones(len(years)) * 110.0})
        # From future of hydrogen
        self.transport_cost = pd.DataFrame(
            {GlossaryCore.Years: years, 'transport': np.ones(len(years)) * 7.6})

        red_meat_daily_cal = np.linspace(46.82, 46.82, year_range)
        white_meat_daily_cal = np.linspace(181.71, 181.71, year_range)
        vegetables_and_carbs_calories_per_day = np.linspace(1774.12, 1774.12, year_range)
        milk_and_eggs = np.linspace(237.98, 237.98, year_range)
        fish_daily_cal = np.linspace(39.02, 39.02, year_range)
        other_food_daily_cal = np.linspace(647.77, 647.77, year_range)
        self.red_meat_calories_per_day = pd.DataFrame({
            GlossaryCore.Years: years,
            'red_meat_calories_per_day': red_meat_daily_cal})
        self.white_meat_calories_per_day = pd.DataFrame({
            GlossaryCore.Years: years,
            'white_meat_calories_per_day': white_meat_daily_cal})
        self.veg_calories_per_day = pd.DataFrame({
            GlossaryCore.Years: years,
            'vegetables_and_carbs_calories_per_day': vegetables_and_carbs_calories_per_day})

        self.milk_eggs_calories_per_day = pd.DataFrame({
            GlossaryCore.Years: years,
            'milk_and_eggs_calories_per_day': milk_and_eggs})

        self.fish_calories_per_day = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.FishDailyCal: fish_daily_cal})
        self.other_food_calories_per_day = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.OtherDailyCal: other_food_daily_cal})

        # bioenergyeurope.org : Dedicated energy crops
        # represent 0.1% of the total biomass production in 2018
        energy_crop_percentage = 0.005
        # ourworldindata, average cereal yield: 4070kg/ha +
        # average yield of switchgrass on grazing lands: 2565,67kg/ha
        # residue is 0.25 more than that
        density_per_ha = 2903 * 1.25
        # available ha of crop: 4.9Gha, initial prod = crop energy + residue for
        # energy of all surfaces
        self.initial_production = 4.8 * density_per_ha * 3.6 * energy_crop_percentage  # in TWh

    def analytic_grad_entry(self):
        return [
            self.test_agriculture_discipline_analytic_grad
        ]

    def test_agriculture_discipline_analytic_grad(self):

        self.model_name = 'crop'
        ns_dict = {'ns_public': f'{self.name}',
                   GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   'ns_biomass_dry': f'{self.name}',
                   'ns_land_use': f'{self.name}',
                   'ns_crop': f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   'ns_invest': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.crop.crop_disc.CropDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{self.model_name}.diet_df': self.diet_df,
                       f'{self.name}.{self.model_name}.kg_to_kcal_dict': self.default_kg_to_kcal,
                       f'{self.name}.{self.model_name}.kg_to_m2_dict': self.default_kg_to_m2,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                       f'{self.name}.{GlossaryCore.TemperatureDfValue}': self.temperature_df,
                       f'{self.name}.red_meat_calories_per_day': self.red_meat_calories_per_day,
                       f'{self.name}.white_meat_calories_per_day': self.white_meat_calories_per_day,
                       f'{self.name}.vegetables_and_carbs_calories_per_day': self.veg_calories_per_day,
                       f'{self.name}.milk_and_eggs_calories_per_day': self.milk_eggs_calories_per_day,
                       f'{self.name}.{GlossaryCore.FishDailyCal}': self.fish_calories_per_day,
                       f'{self.name}.{GlossaryCore.OtherDailyCal}': self.other_food_calories_per_day,
                       f'{self.name}.crop_investment': self.crop_investment,
                       f'{self.name}.margin': self.margin,
                       f'{self.name}.transport_margin': self.margin,
                       f'{self.name}.transport_cost': self.transport_cost,
                       f'{self.name}.data_fuel_dict': BiomassDry.data_energy_dict
                       }
        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_crop_discipline.pkl', discipline=disc_techno, local_data=disc_techno.local_data,
                            step=1e-15, derr_approx='complex_step',
                            inputs=[f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.TemperatureDfValue}',
                                    f'{self.name}.red_meat_calories_per_day',
                                    f'{self.name}.white_meat_calories_per_day',
                                    f'{self.name}.vegetables_and_carbs_calories_per_day',
                                    f'{self.name}.{GlossaryCore.FishDailyCal}',
                                    f'{self.name}.{GlossaryCore.OtherDailyCal}',
                                    f'{self.name}.milk_and_eggs_calories_per_day',
                                    f'{self.name}.crop_investment',
                                    ],
                            outputs=[f'{self.name}.total_food_land_surface',
                                     f'{self.name}.land_use_required',
                                     f'{self.name}.techno_prices',
                                     f'{self.name}.techno_production',
                                     f'{self.name}.techno_consumption',
                                     f'{self.name}.{GlossaryCore.TechnoConsumptionWithoutRatioValue}',
                                     f'{self.name}.CO2_emissions',
                                     f'{self.name}.CO2_land_emission_df',
                                     f'{self.name}.CH4_land_emission_df',
                                     f'{self.name}.N2O_land_emission_df',
                                     f'{self.name}.calories_per_day_constraint',
                                     f'{self.name}.{GlossaryCore.CaloriesPerCapitaValue}'
                                    ])
