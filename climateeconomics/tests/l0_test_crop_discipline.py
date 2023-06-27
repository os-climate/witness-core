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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
import unittest
from os.path import join, dirname
from pandas import read_csv
from climateeconomics.core.core_agriculture.crop import Crop
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from climateeconomics.sos_wrapping.sos_wrapping_agriculture.crop.crop_disc import CropDiscipline
from pathlib import Path

import numpy as np
import pandas as pd


class CropTestCase(unittest.TestCase):

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.year_start = 2020
        self.year_end = 2055
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1

        population = np.array(np.linspace(7800, 9200, year_range))
        self.population_df = pd.DataFrame(
            {"years": years, "population": population})
        self.population_df.index = years
        temperature = np.array(np.linspace(0.0,5.0, year_range))
        self.temperature_df = pd.DataFrame({"years": years, "temp_atmo": temperature})
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
                                 'other': 21.4
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
                            'years': years,
                            'red_meat_percentage': red_meat_percentage})
        self.white_meat_percentage = pd.DataFrame({
                                'years': years,
                                'white_meat_percentage': white_meat_percentage})
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
        self.initial_production = 4.8 * density_per_ha * 3.6 * energy_crop_percentage   # in TWh

        # Our World in Data
        # https://ourworldindata.org/environmental-impacts-of-food#co2-and-greenhouse-gas-emissions

        co2_gwp_100 = 1.0
        ch4_gwp_100 = 28.0
        n2o_gwp_100 = 265.0

        ghg_emissions_unit = 'kg/kg'  # in kgCo2eq per kg of food
        default_ghg_emissions = {'red meat': 32.7,
                                 'white meat': 4.09,
                                 'milk': 1.16,
                                 'eggs': 1.72,
                                 'rice and maize': 1.45,
                                 'potatoes': 0.170,
                                 'fruits and vegetables': 0.372,
                                 'other': 3.44,
                                 }

        # Our World in Data
        # https://ourworldindata.org/carbon-footprint-food-methane
        ch4_emissions_unit = 'kg/kg'  # in kgCH4 per kg food
        calibration = 0.134635 / 0.108958
        # set up as a ratio of total ghg emissions
        ch4_emissions_ratios = {'red meat': 49 / 100 * calibration,
                                'white meat': 2 / 20 * calibration,
                                'milk': 17.0 / 33 * calibration,
                                'eggs': 0.0 * calibration,
                                'rice and maize': 4 / 6.5 * calibration,
                                'potatoes': 0.0 * calibration,
                                'fruits and vegetables': 0.0 * calibration,  # negligible methane in this category
                                'other': (0.0 + 0.0 + 11.0 + 4.0 + 5.0 + 17.0) / (
                                            14 + 24 + 33 + 27 + 29 + 34) * calibration,
                                }

        default_ch4_emissions = {}
        for food in default_ghg_emissions:
            default_ch4_emissions[food] = (default_ghg_emissions[food] * ch4_emissions_ratios[food] / ch4_gwp_100)

        # FAO Stats
        # https://www.fao.org/faostat/en/#data/GT
        n2o_emissions_unit = 'kg/kg'  # in kgN2O per kg food$
        calibration = 7.332 / 6.0199
        pastures_emissions = 3.039e-3 * calibration
        crops_emissions = 1.504e-3 * calibration

        # with land use ratio on n2o emissions
        default_n2o_emissions = {'red meat': pastures_emissions * 3.0239241372696104 / 0.9959932034220041,
                                 'white meat': pastures_emissions * 0.3555438662130599 / 0.9959932034220041,
                                 'milk': pastures_emissions * 0.5564085980770741 / 0.9959932034220041,
                                 'eggs': pastures_emissions * 0.048096212128271996 / 0.9959932034220041,
                                 'rice and maize': crops_emissions * 0.2236252183903196 / 0.29719264680276536,
                                 'potatoes': crops_emissions * 0.023377379498821543 / 0.29719264680276536,
                                 'fruits and vegetables': crops_emissions * 0.13732524416192043 / 0.29719264680276536,
                                 'other': crops_emissions * 0.8044427451599999 / 0.29719264680276536,
                                 }

        # co2 emissions = (total_emissions - ch4_emissions * ch4_gwp_100 - n2o_emissions * n2o_gwp_100)/co2_gwp_100
        co2_emissions_unit = 'kg/kg'  # in kgCo2 per kg food
        calibration = 0.722 / 3.417569
        default_co2_emissions = {'red meat': 0.0 * calibration,
                                 'white meat': 0.0 * calibration,
                                 'milk': 0.0 * calibration,
                                 'eggs': 0.0 * calibration,
                                 'rice and maize': 1.45 * calibration,
                                 'potatoes': 0.170 * calibration,
                                 'fruits and vegetables': 0.372 * calibration,
                                 'other': 3.44 * calibration,
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


        self.param = {'year_start': self.year_start,
                      'year_end': self.year_end,
                      'time_step': self.time_step,
                      'diet_df': self.diet_df,
                      'kg_to_kcal_dict': self.default_kg_to_kcal,
                      'population_df': self.population_df,
                      'temperature_df': self.temperature_df,
                      'kg_to_m2_dict': self.default_kg_to_m2,
                      'other_use_crop': self.other,
                      'param_a':  - 0.00833,
                      'param_b': - 0.04167,
                      'crop_investment': self.crop_investment,
                      'margin': self.margin,
                      'transport_margin': self.margin,
                      'transport_cost': self.transport_cost,
                      'data_fuel_dict': BiomassDry.data_energy_dict,
                      'techno_infos_dict': CropDiscipline.techno_infos_dict_default,
                      'scaling_factor_crop_investment': 1e3,
                      'scaling_factor_techno_consumption': 1e3,
                      'scaling_factor_techno_production': 1e3,
                      'initial_age_distrib': initial_age_distribution,
                      'initial_production': self.initial_production,
                      'red_meat_calories_per_day': self.red_meat_percentage,
                      'white_meat_calories_per_day': self.white_meat_percentage,
                      'vegetables_and_carbs_calories_per_day': self.veg_calories_per_day,
                      'milk_and_eggs_calories_per_day': self.milk_eggs_calories_per_day,
                      'co2_emissions_per_kg': default_co2_emissions,
                      'ch4_emissions_per_kg': default_ch4_emissions,
                      'n2o_emissions_per_kg': default_n2o_emissions,
                      'constraint_calories_limit': 1700. ,
                      'constraint_calories_ref': 3400.
                      }

    def test_crop_model(self):
        ''' 
        Basic test of crop model
        Check the overal run without value checks (will be done in another test)
        '''

        crop = Crop(self.param)
        crop.configure_parameters_update(self.param)
        crop.compute()

    def test_crop_discipline(self):
        '''
        Check discipline setup and run
        '''

        name = 'Test'
        model_name = 'crop'
        ee = ExecutionEngine(name)
        ns_dict = {'ns_public': f'{name}',
                   'ns_ref': f'{name}',
                   'ns_witness': f'{name}.{model_name}',
                   'ns_functions': f'{name}.{model_name}',
                   'ns_agriculture': f'{name}.{model_name}',
                   'ns_biomass_dry': f'{name}.{model_name}',
                   'ns_land_use':f'{name}.{model_name}',
                   'ns_crop':f'{name}.{model_name}',
                   'ns_invest':f'{name}.{model_name}'}

        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.crop.crop_disc.CropDiscipline'
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        inputs_dict = {f'{name}.year_start': self.year_start,
                       f'{name}.year_end': self.year_end,
                       f'{name}.time_step': 1,
                       f'{name}.{model_name}.{Crop.DIET_DF}': self.diet_df,
                       f'{name}.{model_name}.{Crop.KG_TO_KCAL_DICT}': self.default_kg_to_kcal,
                       f'{name}.{model_name}.{Crop.KG_TO_M2_DICT}': self.default_kg_to_m2,
                       f'{name}.{model_name}.{Crop.POPULATION_DF}': self.population_df,
                       f'{name}.{model_name}.red_meat_calories_per_day': self.red_meat_percentage,
                       f'{name}.{model_name}.white_meat_calories_per_day': self.white_meat_percentage,
                       f'{name}.{model_name}.vegetables_and_carbs_calories_per_day': self.veg_calories_per_day,
                       f'{name}.{model_name}.milk_and_eggs_calories_per_day': self.milk_eggs_calories_per_day,
                       f'{name}.{model_name}.{Crop.OTHER_USE_CROP}': self.other,
                       f'{name}.{model_name}.temperature_df': self.temperature_df,
                       f'{name}.{model_name}.crop_investment': self.crop_investment,
                       f'{name}.{model_name}.margin': self.margin,
                       f'{name}.{model_name}.transport_margin': self.margin,
                       f'{name}.{model_name}.transport_cost': self.transport_cost,
                       f'{name}.{model_name}.data_fuel_dict': BiomassDry.data_energy_dict
                       }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(
            f'{name}.{model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        # for graph in graph_list:
        #     graph.to_plotly().show()
