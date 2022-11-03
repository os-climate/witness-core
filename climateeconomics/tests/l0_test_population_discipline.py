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
from scipy.interpolate import interp1d
import pickle
import time
import cProfile
from _io import StringIO
import pstats


class PopDiscTest(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        self.model_name = 'population'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.population.population_discipline.PopulationDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

    def test_execute(self):

        data_dir = join(dirname(__file__), 'data')

        # Test With a GDP that grows at 2%
        years = np.arange(2020, 2101, 1)
        nb_per = 2101 - 2020
        gdp_year_start = 130.187
        gdp_serie = []
        gdp_serie.append(gdp_year_start)
        for year in np.arange(1, nb_per):
            gdp_serie.append(gdp_serie[year - 1] * 1.02)

        economics_df_y = pd.DataFrame(
            {'years': years, 'output_net_of_d': gdp_serie})
        economics_df_y.index = years
        temperature_df_all = read_csv(
            join(data_dir, 'temperature_data_onestep.csv'))

        values_dict = {f'{self.name}.year_start': 2020,
                       f'{self.name}.year_end': 2100,
                       f'{self.name}.economics_df': economics_df_y,
                       f'{self.name}.temperature_df': temperature_df_all
                       }

        self.ee.dm.set_values_from_dict(values_dict)
        t0 = time.time()
        self.ee.execute()
        print('old_time : 8.636150598526001  s ')
        print('Time : ', time.time() - t0, ' s')

        res_pop = self.ee.dm.get_value(f'{self.name}.population_df')

        birth_rate = self.ee.dm.get_value(
            f'{self.name}.{self.model_name}.birth_rate_df')
        life_expectancy_df = self.ee.dm.get_value(
            f'{self.name}.{self.model_name}.life_expectancy_df')

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
#         for graph in graph_list:
#             graph.to_plotly().show()

    def test_economicdegrowth(self):

        data_dir = join(dirname(__file__), 'data')

        # Test With a GDP that grows at 2%
        years = np.arange(2020, 2101, 1)
        nb_per = 2101 - 2020
        gdp_year_start = 130.187
        gdp_serie = []
        temp_serie = []
        gdp_serie.append(gdp_year_start)
        temp_serie.append(0.85)
        for year in np.arange(1, nb_per):
            gdp_serie.append(gdp_serie[year - 1] * 1.02)
            temp_serie.append(temp_serie[year - 1] * 1.01)

        economics_df_y = pd.DataFrame(
            {'years': years, 'output_net_of_d': gdp_serie})
        economics_df_y.index = years
        temperature_df = pd.DataFrame(
            {'years': years, 'temp_atmo': temp_serie})
        temperature_df.index = years

        values_dict = {f'{self.name}.year_start': 2020,
                       f'{self.name}.year_end': 2100,
                       f'{self.name}.economics_df': economics_df_y,
                       f'{self.name}.temperature_df': temperature_df
                       }

        self.ee.dm.set_values_from_dict(values_dict)

        self.ee.execute()

        res_pop = self.ee.dm.get_value(f'{self.name}.population_df')
#        print(res_pop)

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        # for graph in graph_list:
        #     graph.to_plotly().show()

    def test_kcaldegrowth(self):

        # Test With a GDP that grows at 2%
        years = np.arange(2020, 2101, 1)
        nb_per = 2101 - 2020
        gdp_year_start = 130.187
        gdp_serie = []
        temp_serie = []
        gdp_serie.append(gdp_year_start)
        temp_serie.append(0.85)
        for year in np.arange(1, nb_per):
            gdp_serie.append(gdp_serie[year - 1] * 1.02)
            temp_serie.append(temp_serie[year - 1] * 1.01)

        economics_df_y = pd.DataFrame(
            {'years': years, 'output_net_of_d': gdp_serie})
        economics_df_y.index = years
        temperature_df = pd.DataFrame(
            {'years': years, 'temp_atmo': temp_serie})
        temperature_df.index = years
        # Test With a average calorie intake at 2000 kcal per capita
        calories_pc_df = pd.DataFrame(
            {'years': years, 'kcal_pc': np.linspace(2000,2000,len(years))})
        calories_pc_df.index = years

        values_dict = {f'{self.name}.year_start': 2020,
                       f'{self.name}.year_end': 2100,
                       f'{self.name}.economics_df': economics_df_y,
                       f'{self.name}.temperature_df': temperature_df,
                       f'{self.name}.calories_pc_df': calories_pc_df
                       }

        self.ee.dm.set_values_from_dict(values_dict)

        self.ee.execute()

        res_pop = self.ee.dm.get_value(f'{self.name}.population_df')
#        print(res_pop)

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        for graph in graph_list:
            graph.to_plotly().show()


#     def test_ssps_scenario(self):
#
#         data_dir = join(dirname(__file__), 'data')
#
#         gdp_df = read_csv(join(data_dir, 'ssps_gdp.csv'))
#
#         scenario = 'SSP5'
#         years =  np.arange(2020, 2101, 1)
#         f2 = interp1d(gdp_df['years'], gdp_df[scenario])
#         gdp_full = f2(years)
#         economics_df = pd.DataFrame(
#             {'years': years, 'output_net_of_d': gdp_full })
#         economics_df.index = years
#         temperature_df_all = read_csv(
#             join(data_dir, 'temperature_data_onestep.csv'))
#
#         values_dict = {f'{self.name}.year_start': 2020,
#                        f'{self.name}.year_end': 2100,
#                        f'{self.name}.economics_df': economics_df,
#                        f'{self.name}.temperature_df': temperature_df_all
#                        }
#
#         self.ee.dm.set_values_from_dict(values_dict)
#
#         self.ee.execute()
#
#         res_pop = self.ee.dm.get_value(f'{self.name}.population_df')
# #        print(res_pop)
#
#         disc = self.ee.dm.get_disciplines_with_name(
#             f'{self.name}.{self.model_name}')[0]
#         filter = disc.get_chart_filter_list()
#         graph_list = disc.get_post_processing_list(filter)
# #         for graph in graph_list:
# #             graph.to_plotly().show()
#
if '__main__' == __name__:

    cls = PopDiscTest()
    cls.setUp()
    cls.test_execute()
