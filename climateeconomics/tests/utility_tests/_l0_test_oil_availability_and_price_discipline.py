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
from climateeconomics.core.core_resources.oil_availability_and_price_model.oil_availability_and_price_prediction_model import dataset, reserves_model, price_model
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine

import numpy as np
import pandas as pd


class OilModelTestCase(unittest.TestCase):

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.year_start = 2020
        self.year_end = 2100

        self.param = {'year_start': self.year_start,
                      'year_end': self.year_end}

        self.dt = dataset()
        self.res = reserves_model(self.dt, self.param)
        self.pri = price_model(self.dt, self.res, self.param)

    def test_oil_model(self):
        '''
        Basique test of land use model
        Mainly check the overal run without value checks (will be done in another test)
        '''   
        self.res.world_reserves()
        self.res.plot_world_reserves_vs_country_shortage()
        self.pri.plot_baseline()
        self.pri.plot_price_vs_prod_change()
        self.pri.plot_price_preds()
        

    def test_oil_discipline(self):
        ''' 
        Check discipline setup and run
        '''

        name = 'Test'
        model_name = 'oil_use'
        ee = ExecutionEngine(name)
        ns_dict = {'ns_public': f'{name}',
                   'ns_witness': f'{name}.{model_name}',
                   'ns_functions': f'{name}.{model_name}',
                   'oil': f'{name}.{model_name}',
                   'oil_availability_and_price': f'{name}.{model_name}.oil_availability_and_price',}
        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_resources.sos_wrapping_oil_resource.oil_availability_and_price_model.oil_availability_and_price_disc.OilDiscipline'
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        inputs_dict = {f'{name}.year_start': self.year_start,
                       f'{name}.year_end': self.year_end,
                       }
        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(
            f'{name}.{model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
#        for graph in graph_list:
#            graph.to_plotly().show()
    
if __name__ == '__main__':
    oil_test = OilModelTestCase()
    oil_test.setUp()
    # oil_test.test_oil_model()
    oil_test.test_oil_discipline()
