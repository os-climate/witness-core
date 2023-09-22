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
import unittest
import numpy as np
import pandas as pd
from os.path import join, dirname
from pandas import DataFrame, read_csv

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class CarbonCycleDiscTest(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_execute(self):

        self.model_name = 'carboncycle'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.carboncycle.carboncycle_discipline.CarbonCycleDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')

        CO2_emissions_df_all = read_csv(
            join(data_dir, 'co2_emissions_onestep.csv'))

        CO2_emissions_df_y = CO2_emissions_df_all[CO2_emissions_df_all[GlossaryCore.Years] >= 2020]

        # put manually the index
        years = np.arange(2020, 2101)
        CO2_emissions_df_y.index = years

        values_dict = {f'{self.name}.CO2_emissions_df': CO2_emissions_df_y}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        res_carbon_cycle = self.ee.dm.get_value(f'{self.name}.{GlossaryCore.CarbonCycleDfValue}')

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
#         for graph in graph_list:
#             graph.to_plotly().show()
