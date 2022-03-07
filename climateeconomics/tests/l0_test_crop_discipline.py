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
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
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

        population = np.array(np.linspace(7800, 7800, year_range))
        self.population_df = pd.DataFrame(
            {"years": years, "population": population})
        self.population_df.index = years
        temperature = np.array(np.linspace(1.05, 5, year_range))
        self.temperature_df = pd.DataFrame(
            {"years": years, "temp_atmo": temperature})
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
        self.red_to_white_meat = np.linspace(0, 50, year_range)
        self.meat_to_vegetables = np.linspace(0, 100, year_range)

        default_qauntity_1 = np.array(np.linspace(30, 2, year_range))
        default_qauntity_2 = np.array(np.linspace(10, 30, year_range))

        self.diet_df = pd.DataFrame({'red meat': [11.02],
                                     'white meat': [31.11],
                                     'milk': [79.27],
                                     'eggs': [9.68],
                                     'rice and maize': [97.76],
                                     'potatoes': [32.93],
                                     'fruits and vegetables': [217.62],
                                     })
        self.other = np.array(np.linspace(0.102, 0.102, year_range))

        self.param = {'year_start': self.year_start,
                      'year_end': self.year_end,
                      'time_step': self.time_step,
                      'diet_df': self.diet_df,
                      'kg_to_kcal_dict': self.default_kg_to_kcal,
                      'population_df': self.population_df,
                      'temperature_df': self.temperature_df,
                      'kg_to_m2_dict': self.default_kg_to_m2,
                      'red_to_white_meat': self.red_to_white_meat,
                      'meat_to_vegetables': self.meat_to_vegetables,
                      'other_use_crop': self.other,
                      'param_a':  - 0.00833,
                      'param_b': - 0.04167
                      }

    def test_crop_model(self):
        ''' 
        Basic test of crop model
        Check the overal run without value checks (will be done in another test)
        '''

        crop = Crop(self.param)

        crop.compute(self.population_df, self.temperature_df)

    def test_crop_discipline(self):
        '''
        Check discipline setup and run
        '''

        name = 'Test'
        model_name = 'crop'
        ee = ExecutionEngine(name)
        ns_dict = {'ns_public': f'{name}',
                   'ns_witness': f'{name}.{model_name}',
                   'ns_functions': f'{name}.{model_name}',
                   'ns_agriculture': f'{name}.{model_name}'}
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
                       f'{name}.{model_name}.{Crop.RED_TO_WHITE_MEAT}': self.red_to_white_meat,
                       f'{name}.{model_name}.{Crop.MEAT_TO_VEGETABLES}': self.meat_to_vegetables,
                       f'{name}.{model_name}.{Crop.OTHER_USE_CROP}': self.other,
                       f'{name}.{model_name}.temperature_df': self.temperature_df
                       }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(
            f'{name}.{model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        for graph in graph_list:
            graph.to_plotly().show()
