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
from os.path import join, dirname

import numpy as np
from pandas import read_csv

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class IndusEmissionDiscTest(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_execute(self):

        self.model_name = 'indusemission'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   GlossaryCore.NS_CCS: f'{self.name}',
                   'ns_energy': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_emissions.indus_emissions.indusemissions_discipline.IndusemissionsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')

        economics_df_all = read_csv(
            join(data_dir, 'economics_data_onestep.csv'))
        energy_supply_df_all = read_csv(
            join(data_dir, 'energy_supply_data_onestep.csv'))

        economics_df_y = economics_df_all[economics_df_all[GlossaryCore.Years] >= GlossaryCore.YeartStartDefault]
        energy_supply_df_y = energy_supply_df_all[energy_supply_df_all[GlossaryCore.Years] >= GlossaryCore.YeartStartDefault]
        energy_supply_df_y = energy_supply_df_y.rename(
            columns={'total_CO2_emitted': GlossaryCore.TotalCO2Emissions})

        # put manually the index
        years = np.arange(GlossaryCore.YeartStartDefault, GlossaryCore.YeartEndDefault +1)
        economics_df_y.index = years
        energy_supply_df_y.index = years

        values_dict = {f'{self.name}.{GlossaryCore.EconomicsDfValue}': economics_df_y,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CheckRangeBeforeRunBoolName}': False,
                       }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        res_indus_emissions = self.ee.dm.get_value(
            f'{self.name}.CO2_indus_emissions_df')
        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
#         for graph in graph_list:
#             graph.to_plotly().show()
