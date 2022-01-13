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
from climateeconomics.core.core_deforest.deforest import Deforest
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from pathlib import Path

import numpy as np
import pandas as pd


class DeforestationTestCase(unittest.TestCase):

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.year_start = 2020
        self.year_end = 2055
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        forest_surface = np.array(np.linspace(4, 2.8, year_range))
        self.forest_df = pd.DataFrame(
            {"years": years, "forest_surface": forest_surface})
        deforestation_rate = np.array(np.linspace(1, 0.2, year_range))
        self.deforestation_rate_df = pd.DataFrame(
            {"years": years, "forest_evolution": deforestation_rate})
        self.CO2_per_ha = 4000

        self.param = {'year_start': self.year_start,
                      'year_end': self.year_end,
                      'time_step': self.time_step,
                      Deforest.FOREST_DF: self.forest_df,
                      Deforest.DEFORESTATION_RATE_DF: self.deforestation_rate_df,
                      Deforest.CO2_PER_HA: self.CO2_per_ha,
                      }

    def test_forest_model(self):
        ''' 
        Basique test of deforestation model
        Mainly check the overal run without value checks (will be done in another test)
        '''

        deforestation = Deforest(self.param)

        deforestation.compute(self.param)

    def test_forest_discipline(self):
        '''
        Check discipline setup and run
        '''

        name = 'Test'
        model_name = 'deforestation'
        ee = ExecutionEngine(name)
        ns_dict = {'ns_public': f'{name}',
                   'ns_witness': f'{name}.{model_name}',
                   'ns_functions': f'{name}.{model_name}',
                   'ns_deforestation': f'{name}.{model_name}'}
        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_deforest.deforest.deforest_disc.DeforestDiscipline'
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        inputs_dict = {f'{name}.year_start': self.year_start,
                       f'{name}.year_end': self.year_end,
                       f'{name}.time_step': 1,
                       f'{name}.{model_name}.{Deforest.FOREST_DF}': self.forest_df,
                       f'{name}.{model_name}.{Deforest.DEFORESTATION_RATE_DF}': self.deforestation_rate_df,
                       f'{name}.{model_name}.{Deforest.CO2_PER_HA}': self.CO2_per_ha,
                       }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(
            f'{name}.{model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        for graph in graph_list:
            graph.to_plotly().show()
