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


class DamageDiscTest(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_execute(self):

        self.model_name = 'damage'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.damagemodel.damagemodel_discipline.DamageDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')

        economics_df_all = read_csv(
            join(data_dir, 'economics_data_onestep.csv'))
        temperature_df_all = read_csv(
            join(data_dir, 'temperature_data_onestep.csv'))

        economics_df_all = economics_df_all[GlossaryCore.EconomicsDf['dataframe_descriptor'].keys()]

        economics_df_y = economics_df_all[economics_df_all[GlossaryCore.Years] >= 2020]
        temperature_df_y = temperature_df_all[temperature_df_all[GlossaryCore.Years] >= 2020]

        years = np.arange(2020, 2101, 1)
        economics_df_y.index = years
        temperature_df_y.index = years

        values_dict = {f'{self.name}.{self.model_name}.tipping_point': True,
                       f'{self.name}.{GlossaryCore.EconomicsDfValue}': economics_df_y,
                       f'{self.name}.{GlossaryCore.TemperatureDfValue}': temperature_df_y,
                       f'{self.name}.total_emissions_ref': 37.,
                       f'{self.name}.{self.model_name}.damage_constraint_factor': np.concatenate((np.linspace(0.5, 1, 15), np.asarray([1] * (len(years) - 15))))
                       }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        res_damage = self.ee.dm.get_value(f'{self.name}.{GlossaryCore.DamageDfValue}')

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)

        # for graph in graph_list:
        #    graph.to_plotly().show()
