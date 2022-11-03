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

from climateeconomics.core.core_agriculture.crop import Crop
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from climateeconomics.sos_wrapping.sos_wrapping_agriculture.crop.crop_disc import CropDiscipline
import unittest
import pandas as pd
import numpy as np


class AgricultureJacobianDiscTest(AbstractJacobianUnittest):
    AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        '''
        Initialize third data needed for testing
        '''
        self.year_start = 2020
        self.year_end = 2055
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1

        population = np.array(np.linspace(7800, 7800, year_range))
        self.population_df = pd.DataFrame(
            {"years": years, "population": population})
        self.population_df.index = years
        temperature = np.array(np.linspace(1.05, 5, year_range))
        self.temperature_df = pd.DataFrame(
            {"years": years, "temp_atmo": temperature})
        self.temperature_df.index = years

        lifetime = 50

        # Age distribution of forests in 2008 (
        initial_age_distribution = pd.DataFrame({'age': np.arange(1, lifetime),
                                                 'distrib': [0.16, 0.24, 0.31, 0.39, 0.47, 0.55, 0.63, 0.71, 0.78, 0.86,
                                                             0.94, 1.02, 1.1, 1.18, 1.26, 1.33, 1.41, 1.49, 1.57, 1.65,
                                                             1.73, 1.81, 1.88, 1.96, 2.04, 2.12, 2.2, 2.28, 2.35, 2.43,
                                                             2.51, 2.59, 2.67, 2.75, 2.83, 2.9, 2.98, 3.06, 3.14, 3.22,
                                                             3.3, 3.38, 3.45, 3.53, 3.61, 3.69, 3.77, 3.85, 3.92]})

        self.default_kg_to_m2 = {'red meat': 360,
                                 'white meat': 16,
                                 'milk': 8.95,
                                 'eggs': 6.3,
                                 'rice and maize': 2.9,
                                 'potatoes': 0.88,
                                 'fruits and vegetables': 0.8,
                                 'other': 21.4,
                                 }
        self.default_kg_to_kcal = {'red meat': 2566,
                                   'white meat': 1860,
                                   'milk': 550,
                                   'eggs': 1500,
                                   'rice and maize': 1150,
                                   'potatoes': 670,
                                   'fruits and vegetables': 624,
                                   }
        red_meat_percentage = np.linspace(600, 700, year_range)
        white_meat_percentage = np.linspace(700, 600, year_range)
        vegetables_and_carbs_calories_per_day = np.linspace(800, 1200, year_range)
        self.red_meat_percentage = pd.DataFrame({
                            'years': years,
                            'red_meat_calories_per_day': red_meat_percentage})
        self.white_meat_percentage = pd.DataFrame({
                                'years': years,
                                'white_meat_calories_per_day': white_meat_percentage})
        self.veg_calories_per_day = pd.DataFrame({
                                'years': years,
                                'vegetables_and_carbs_calories_per_day': vegetables_and_carbs_calories_per_day})

        self.milk_eggs_calories_per_day = pd.DataFrame({
                                'years': years,
                                'milk_and_eggs_calories_per_day': vegetables_and_carbs_calories_per_day})

        self.diet_df = pd.DataFrame({'red meat': [11.02],
                                     'white meat': [31.11],
                                     'milk': [79.27],
                                     'eggs': [9.68],
                                     'rice and maize': [97.76],
                                     'potatoes': [32.93],
                                     'fruits and vegetables': [217.62],
                                     })
        self.other = np.array(np.linspace(0.102, 0.102, year_range))

        # investment: 1Mha of crop land each year
        self.crop_investment = pd.DataFrame(
            {'years': years, 'investment': np.ones(len(years)) * 0.381})

        self.margin = pd.DataFrame(
            {'years': years, 'margin': np.ones(len(years)) * 110.0})

        # From future of hydrogen
        self.transport_cost = pd.DataFrame(
            {'years': years, 'transport': np.ones(len(years)) * 7.6})


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

        self.param = {'year_start': self.year_start,
                      'year_end': self.year_end,
                      'time_step': self.time_step,
                      'diet_df': self.diet_df,
                      'kg_to_kcal_dict': self.default_kg_to_kcal,
                      'population_df': self.population_df,
                      'temperature_df': self.temperature_df,
                      'kg_to_m2_dict': self.default_kg_to_m2,
                      'red_meat_percentage': self.red_meat_percentage,
                      'white_meat_percentage': self.white_meat_percentage,
                      'other_use_crop': self.other,
                      'param_a': - 0.00833,
                      'param_b': - 0.04167,
                      'crop_investment': self.crop_investment,
                      'margin': self.margin,
                      'transport_margin': self.margin,
                      'transport_cost': self.transport_cost,
                      'data_fuel_dict': BiomassDry.data_energy_dict,
                      'techno_infos_dict': CropDiscipline.techno_infos_dict_default,
                      'scaling_factor_crop_investment': 1e3,
                      'initial_age_distrib': initial_age_distribution,
                      'initial_production': self.initial_production
                      }

    def analytic_grad_entry(self):
        return [
            self.test_agriculture_discipline_analytic_grad
        ]

    def test_agriculture_discipline_analytic_grad(self):

        self.model_name = 'crop'
        ns_dict = {'ns_public': f'{self.name}',
                   'ns_witness': f'{self.name}',
                   'ns_functions': f'{self.name}',
                   'ns_biomass_dry': f'{self.name}',
                   'ns_land_use':f'{self.name}',
                   'ns_crop':f'{self.name}',
                   'ns_ref': f'{self.name}',
                   'ns_invest':f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.crop.crop_disc.CropDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()


        values_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.{self.model_name}.diet_df': self.diet_df,
                       f'{self.name}.{self.model_name}.kg_to_kcal_dict': self.default_kg_to_kcal,
                       f'{self.name}.{self.model_name}.kg_to_m2_dict': self.default_kg_to_m2,
                       f'{self.name}.population_df': self.population_df,
                       f'{self.name}.temperature_df': self.temperature_df,
                       f'{self.name}.red_meat_calories_per_day': self.red_meat_percentage,
                       f'{self.name}.white_meat_calories_per_day': self.white_meat_percentage,
                       f'{self.name}.vegetables_and_carbs_calories_per_day': self.veg_calories_per_day,
                       f'{self.name}.milk_and_eggs_calories_per_day': self.milk_eggs_calories_per_day,
                       f'{self.name}.{self.model_name}.{Crop.OTHER_USE_CROP}': self.other,
                       f'{self.name}.crop_investment': self.crop_investment,
                       f'{self.name}.margin': self.margin,
                       f'{self.name}.transport_margin': self.margin,
                       f'{self.name}.transport_cost': self.transport_cost,
                       f'{self.name}.data_fuel_dict': BiomassDry.data_energy_dict
                       }
        self.ee.dm.set_values_from_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.sos_disciplines[0]
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_crop_discipline.pkl', discipline=disc_techno,
                            step=1e-15, derr_approx='complex_step',
                            inputs=[f'{self.name}.population_df',
                                    f'{self.name}.temperature_df',
                                    f'{self.name}.red_meat_calories_per_day',
                                    f'{self.name}.white_meat_calories_per_day',
                                    f'{self.name}.vegetables_and_carbs_calories_per_day',
                                    f'{self.name}.milk_and_eggs_calories_per_day',
                                    f'{self.name}.crop_investment',
                                    ],
                            outputs=[f'{self.name}.total_food_land_surface',
                                     f'{self.name}.land_use_required',
                                     f'{self.name}.techno_prices',
                                     f'{self.name}.techno_production',
                                     f'{self.name}.techno_consumption',
                                     f'{self.name}.techno_consumption_woratio',
                                     f'{self.name}.CO2_emissions',
                                     f'{self.name}.CO2_land_emission_df',
                                     f'{self.name}.CH4_land_emission_df',
                                     f'{self.name}.N2O_land_emission_df',
                                     f'{self.name}.calories_per_day_constraint',
                                     f'{self.name}.calories_pc_df'
                                    ])
