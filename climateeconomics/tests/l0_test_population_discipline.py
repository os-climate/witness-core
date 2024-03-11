'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/21-2023/11/03 Copyright 2023 Capgemini

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
import time
import unittest
from os.path import join, dirname

import numpy as np
import pandas as pd
from pandas import read_csv

from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class PopDiscTest(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        self.model_name = GlossaryCore.PopulationValue
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
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
        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        nb_per = GlossaryCore.YearEndDefault + 1 - GlossaryCore.YearStartDefault
        gdp_year_start = 130.187
        gdp_serie = []
        gdp_serie.append(gdp_year_start)
        for year in np.arange(1, nb_per):
            gdp_serie.append(gdp_serie[year - 1] * 1.02)

        economics_df_y = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.OutputNetOfDamage: gdp_serie})
        economics_df_y.index = years
        temperature_df_all = read_csv(
            join(data_dir, 'temperature_data_onestep.csv'))

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': GlossaryCore.YearStartDefault,
                       f'{self.name}.{GlossaryCore.YearEnd}': GlossaryCore.YearEndDefault,
                       f'{self.name}.{GlossaryCore.EconomicsDfValue}': economics_df_y,
                       f'{self.name}.{GlossaryCore.TemperatureDfValue}': temperature_df_all,
                       f'{self.name}.{self.model_name}.{GlossaryCore.CheckRangeBeforeRunBoolName}': False,
                       }

        self.ee.load_study_from_input_dict(values_dict)
        t0 = time.time()
        self.ee.execute()
        print('old_time : 8.636150598526001  s ')
        print('Time : ', time.time() - t0, ' s')

        res_pop = self.ee.dm.get_value(f'{self.name}.{GlossaryCore.PopulationDfValue}')

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
        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        nb_per = GlossaryCore.YearEndDefault + 1 - GlossaryCore.YearStartDefault
        gdp_year_start = 130.187
        gdp_serie = []
        temp_serie = []
        gdp_serie.append(gdp_year_start)
        temp_serie.append(0.85)
        for year in np.arange(1, nb_per):
            gdp_serie.append(gdp_serie[year - 1] * 1.02)
            temp_serie.append(temp_serie[year - 1] * 1.01)

        economics_df_y = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.OutputNetOfDamage: gdp_serie})
        economics_df_y.index = years
        temperature_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.TempAtmo: temp_serie})
        temperature_df.index = years

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': GlossaryCore.YearStartDefault,
                       f'{self.name}.{GlossaryCore.YearEnd}': GlossaryCore.YearEndDefault,
                       f'{self.name}.{GlossaryCore.EconomicsDfValue}': economics_df_y,
                       f'{self.name}.{GlossaryCore.TemperatureDfValue}': temperature_df
                       }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        res_pop = self.ee.dm.get_value(f'{self.name}.{GlossaryCore.PopulationDfValue}')
#        print(res_pop)

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        # for graph in graph_list:
        #     graph.to_plotly().show()

    def test_kcaldegrowth(self):

        # Test With a GDP that grows at 2%
        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        nb_per = GlossaryCore.YearEndDefault + 1 - GlossaryCore.YearStartDefault
        gdp_year_start = 130.187
        gdp_serie = []
        temp_serie = []
        gdp_serie.append(gdp_year_start)
        temp_serie.append(0.85)
        for year in np.arange(1, nb_per):
            gdp_serie.append(gdp_serie[year - 1] * 1.02)
            temp_serie.append(temp_serie[year - 1] * 1.01)

        economics_df_y = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.OutputNetOfDamage: gdp_serie})
        economics_df_y.index = years
        temperature_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.TempAtmo: temp_serie})
        temperature_df.index = years
        # Test With a average calorie intake at 2000 kcal per capita
        calories_pc_df = pd.DataFrame(
            {GlossaryCore.Years: years, 'kcal_pc': np.linspace(2000,2000,len(years))})
        calories_pc_df.index = years

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': GlossaryCore.YearStartDefault,
                       f'{self.name}.{GlossaryCore.YearEnd}': GlossaryCore.YearEndDefault,
                       f'{self.name}.{GlossaryCore.EconomicsDfValue}': economics_df_y,
                       f'{self.name}.{GlossaryCore.TemperatureDfValue}': temperature_df,
                       f'{self.name}.{GlossaryCore.CaloriesPerCapitaValue}': calories_pc_df
                       }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        res_pop = self.ee.dm.get_value(f'{self.name}.{GlossaryCore.PopulationDfValue}')
#        print(res_pop)

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        # for graph in graph_list:
        #    graph.to_plotly().show()

    def test_deactivate_climate_effect_flag(self):

        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        nb_per = GlossaryCore.YearEndDefault + 1 - GlossaryCore.YearStartDefault
        gdp_year_start = 130.187
        gdp_serie = []
        temp_serie = []
        gdp_serie.append(gdp_year_start)
        temp_serie.append(0.85)
        for year in np.arange(1, nb_per):
            gdp_serie.append(gdp_serie[year - 1] * 1.02)
            temp_serie.append(temp_serie[year - 1] * 1.01)

        economics_df_y = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.OutputNetOfDamage: gdp_serie})
        economics_df_y.index = years
        temperature_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.TempAtmo: temp_serie})
        temperature_df.index = years
        # Test With a average calorie intake at 2000 kcal per capita
        calories_pc_df = pd.DataFrame(
            {GlossaryCore.Years: years, 'kcal_pc': np.linspace(2000,2000,len(years))})
        calories_pc_df.index = years

        assumptions_dict = ClimateEcoDiscipline.assumptions_dict_default
        assumptions_dict['activate_climate_effect_population'] = False
        assumptions_dict['activate_pandemic_effect_population'] = False

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': GlossaryCore.YearStartDefault,
                       f'{self.name}.{GlossaryCore.YearEnd}': GlossaryCore.YearEndDefault,
                       f'{self.name}.{GlossaryCore.EconomicsDfValue}': economics_df_y,
                       f'{self.name}.{GlossaryCore.TemperatureDfValue}': temperature_df,
                       f'{self.name}.{GlossaryCore.CaloriesPerCapitaValue}': calories_pc_df,
                       f'{self.name}.assumptions_dict': assumptions_dict
                       }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        res_pop = self.ee.dm.get_value(f'{self.name}.{GlossaryCore.PopulationDfValue}')
#        print(res_pop)

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        for graph in graph_list:
           #graph.to_plotly().show()
           pass


if '__main__' == __name__:

    cls = PopDiscTest()
    cls.setUp()
    cls.test_execute()
