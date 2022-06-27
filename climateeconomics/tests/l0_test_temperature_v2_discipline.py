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

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine


class TemperatureDiscTest(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_01_execute_DICE(self):

        self.model_name = 'temperature'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange_v2.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_ally = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))
        # Take only from year start value
        ghg_cycle_df = carboncycle_df_ally[carboncycle_df_ally['years'] >= 2020]

        ghg_cycle_df['co2_ppm'] = ghg_cycle_df['ppm']
        ghg_cycle_df['ch4_ppm'] = ghg_cycle_df['ppm'] * 1222/296
        ghg_cycle_df['n2o_ppm'] = ghg_cycle_df['ppm'] * 296/296
        ghg_cycle_df = ghg_cycle_df[['years', 'co2_ppm', 'ch4_ppm', 'n2o_ppm']]

        # put manually the index
        years = np.arange(2020, 2101, 1)
        ghg_cycle_df.index = years

        values_dict = {f'{self.name}.year_start': 2020,
                       f'{self.name}.year_end': 2100,
                       f'{self.name}.time_step': 1,
                       f'{self.name}.{self.model_name}.temperature_model': 'DICE',
                       f'{self.name}.ghg_cycle_df': ghg_cycle_df}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        temps = self.ee.dm.get_value(f'{self.name}.temperature_df')
        forcs = self.ee.dm.get_value(f'{self.name}.{self.model_name}.forcing_detail_df')

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        # for graph in graph_list:
        #     graph.to_plotly().show()

    def test_02_execute_DICE_Etminan(self):

        self.model_name = 'temperature'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange_v2.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_ally = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))
        # Take only from year start value
        ghg_cycle_df = carboncycle_df_ally[carboncycle_df_ally['years'] >= 2020]

        ghg_cycle_df['co2_ppm'] = ghg_cycle_df['ppm']
        ghg_cycle_df['ch4_ppm'] = ghg_cycle_df['ppm'] * 1222/296
        ghg_cycle_df['n2o_ppm'] = ghg_cycle_df['ppm'] * 296/296
        ghg_cycle_df = ghg_cycle_df[['years', 'co2_ppm', 'ch4_ppm', 'n2o_ppm']]

        # put manually the index
        years = np.arange(2020, 2101, 1)
        ghg_cycle_df.index = years

        values_dict = {f'{self.name}.year_start': 2020,
                       f'{self.name}.year_end': 2100,
                       f'{self.name}.time_step': 1,
                       f'{self.name}.{self.model_name}.temperature_model': 'DICE',
                       f'{self.name}.{self.model_name}.forcing_model': 'Etminan',
                       f'{self.name}.ghg_cycle_df': ghg_cycle_df}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        temps = self.ee.dm.get_value(f'{self.name}.temperature_df')
        forcs = self.ee.dm.get_value(f'{self.name}.{self.model_name}.forcing_detail_df')

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        # for graph in graph_list:
        #     graph.to_plotly().show()

    def test_03_execute_DICE_Meinshausen(self):

        self.model_name = 'temperature'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange_v2.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_ally = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))
        # Take only from year start value
        ghg_cycle_df = carboncycle_df_ally[carboncycle_df_ally['years'] >= 2020]

        ghg_cycle_df['co2_ppm'] = ghg_cycle_df['ppm']
        ghg_cycle_df['ch4_ppm'] = ghg_cycle_df['ppm'] * 1222/296
        ghg_cycle_df['n2o_ppm'] = ghg_cycle_df['ppm'] * 296/296
        ghg_cycle_df = ghg_cycle_df[['years', 'co2_ppm', 'ch4_ppm', 'n2o_ppm']]

        # put manually the index
        years = np.arange(2020, 2101, 1)
        ghg_cycle_df.index = years

        values_dict = {f'{self.name}.year_start': 2020,
                       f'{self.name}.year_end': 2100,
                       f'{self.name}.time_step': 1,
                       f'{self.name}.{self.model_name}.temperature_model': 'DICE',
                       f'{self.name}.{self.model_name}.forcing_model': 'Meinshausen',
                       f'{self.name}.ghg_cycle_df': ghg_cycle_df}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        temps = self.ee.dm.get_value(f'{self.name}.temperature_df')
        forcs = self.ee.dm.get_value(f'{self.name}.{self.model_name}.forcing_detail_df')

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        # for graph in graph_list:
        #     graph.to_plotly().show()

    def test_04_execute_FUND_Mhyre(self):

        self.model_name = 'temperature'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange_v2.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_ally = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))
        # Take only from year start value
        ghg_cycle_df = carboncycle_df_ally[carboncycle_df_ally['years'] >= 2020]

        ghg_cycle_df['co2_ppm'] = ghg_cycle_df['ppm']
        ghg_cycle_df['ch4_ppm'] = ghg_cycle_df['ppm'] * 1222/296
        ghg_cycle_df['n2o_ppm'] = ghg_cycle_df['ppm'] * 296/296
        ghg_cycle_df = ghg_cycle_df[['years', 'co2_ppm', 'ch4_ppm', 'n2o_ppm']]


        # put manually the index
        years = np.arange(2020, 2101, 1)
        ghg_cycle_df.index = years

        values_dict = {f'{self.name}.year_start': 2020,
                       f'{self.name}.year_end': 2100,
                       f'{self.name}.time_step': 1,
                       f'{self.name}.{self.model_name}.temperature_model': 'FUND',
                       f'{self.name}.ghg_cycle_df': ghg_cycle_df}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        res_damage = self.ee.dm.get_value(f'{self.name}.temperature_df')
        forcs = self.ee.dm.get_value(f'{self.name}.{self.model_name}.forcing_detail_df')

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        # for graph in graph_list:
        #     graph.to_plotly().show()

    def test_05_execute_FUND_Meinshausen(self):

        self.model_name = 'temperature'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange_v2.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_ally = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))
        # Take only from year start value
        ghg_cycle_df = carboncycle_df_ally[carboncycle_df_ally['years'] >= 2020]

        ghg_cycle_df['co2_ppm'] = ghg_cycle_df['ppm']
        ghg_cycle_df['ch4_ppm'] = ghg_cycle_df['ppm'] * 1222/296
        ghg_cycle_df['n2o_ppm'] = ghg_cycle_df['ppm'] * 296/296
        ghg_cycle_df = ghg_cycle_df[['years', 'co2_ppm', 'ch4_ppm', 'n2o_ppm']]


        # put manually the index
        years = np.arange(2020, 2101, 1)
        ghg_cycle_df.index = years

        values_dict = {f'{self.name}.year_start': 2020,
                       f'{self.name}.year_end': 2100,
                       f'{self.name}.time_step': 1,
                       f'{self.name}.{self.model_name}.temperature_model': 'FUND',
                       f'{self.name}.{self.model_name}.forcing_model': 'Meinshausen',
                       f'{self.name}.ghg_cycle_df': ghg_cycle_df}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        res_damage = self.ee.dm.get_value(f'{self.name}.temperature_df')
        forcs = self.ee.dm.get_value(f'{self.name}.{self.model_name}.forcing_detail_df')

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        # for graph in graph_list:
        #     graph.to_plotly().show()
