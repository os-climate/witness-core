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
import unittest
from os.path import dirname, join

import numpy as np
from pandas import read_csv
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.glossarycore import GlossaryCore


class TemperatureDiscTest(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_execute(self):

        self.model_name = 'temperature'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_ally = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))
        # Take only from year start value
        carboncycle_df = carboncycle_df_ally[carboncycle_df_ally[GlossaryCore.Years] >= GlossaryCore.YearStartDefault]
        carboncycle_df = carboncycle_df[[GlossaryCore.Years, "atmo_conc"]]

        # put manually the index
        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        carboncycle_df.index = years

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': GlossaryCore.YearStartDefault,
                       f'{self.name}.{GlossaryCore.YearEnd}': GlossaryCore.YearEndDefault,
                       f'{self.name}.{GlossaryCore.CarbonCycleDfValue}': carboncycle_df,
                       }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        res_damage = self.ee.dm.get_value(f'{self.name}.{GlossaryCore.TemperatureDfValue}')

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
#         for graph in graph_list:
#             graph.to_plotly().show()
