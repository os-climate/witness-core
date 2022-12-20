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

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class ConsumptionDiscTest(unittest.TestCase):
    np.set_printoptions(threshold=np.inf)
    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_execute(self):

        self.model_name = 'consumption'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.consumption.consumption_discipline.ConsumptionDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')

        economics_df = read_csv(
            join(data_dir, 'economics_data_onestep.csv'))
        economics_df = economics_df[economics_df['years'] >= 2020]

        global_data_dir = join(dirname(dirname(__file__)), 'data')
        population_df = read_csv(
            join(global_data_dir, 'population_df.csv'))

        # put manually the index
        years = np.arange(2020, 2101, 1)
        economics_df.index = years
        population_df.index = years
        energy_price = np.arange(200, 200 + len(years))
        energy_mean_price = pd.DataFrame(
            {'years': years, 'energy_price': energy_price})
        residential_energy_conso_ref = 21
        residential_energy = np.linspace(21, 15, len(years))
        residential_energy_df = pd.DataFrame(
            {'years': years, 'residential_energy': residential_energy})
        #Share invest
        share_invest = np.asarray([27.0] * len(years))
        total_investment_share_of_gdp = pd.DataFrame({'years':years, 'share_investment': share_invest})
        np.set_printoptions(threshold=np.inf)
        values_dict = {f'{self.name}.year_start': 2020,
                       f'{self.name}.year_end': 2100,
                       f'{self.name}.time_step': 1,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.economics_df': economics_df,
                       f'{self.name}.population_df': population_df,
                       f'{self.name}.energy_mean_price': energy_mean_price,
                       f'{self.name}.residential_energy_conso_ref': residential_energy_conso_ref,
                       f'{self.name}.residential_energy': residential_energy_df,
                       f'{self.name}.total_investment_share_of_gdp': total_investment_share_of_gdp}

        self.ee.dm.set_values_from_dict(values_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        # for graph in graph_list:
        #     graph.to_plotly().show()
