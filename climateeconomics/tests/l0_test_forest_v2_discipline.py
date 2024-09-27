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

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.core.core_forest.forest_v2 import Forest
from climateeconomics.glossarycore import GlossaryCore


class ForestTestCase(unittest.TestCase):

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2050
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        self.CO2_per_ha = 4000
        # GtCO2
        self.initial_emissions = -7.6
        forest_invest = np.linspace(45, 50, year_range)
        self.forest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, "forest_investment": forest_invest})
        self.reforestation_cost_per_ha = 13800

        construction_delay = 3

        self.invest_before_year_start = pd.DataFrame(
            {'past_years': np.arange(-construction_delay, 0),
             GlossaryCore.InvestmentsValue: np.array([1.135081] * construction_delay)})

        mw_invest = np.linspace(10, 10, year_range)
        self.mw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        transport = np.linspace(7.6, 7.6, year_range)
        self.transport_df = pd.DataFrame(
            {GlossaryCore.Years: years, "transport": transport})
        self.margin = pd.DataFrame(
            {GlossaryCore.Years: years, 'margin': np.ones(len(years)) * 110.0})
        self.initial_protected_forest_surface = 4 * 0.21
        self.initial_unmanaged_forest_surface = 4 - \
                                                1.25 - self.initial_protected_forest_surface
        deforest_invest = np.linspace(10, 5, year_range)
        # case over deforestation
        # deforest_invest = np.linspace(1000, 5000, year_range)
        self.deforest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: deforest_invest})

    # def test_forest_discipline_IEA_NZE_calibration(self):
    #     '''
    #     Calibrating forest invests so that the energy production matches the IEA-NZE production
    #     '''
    #
    #     name = 'Test'
    #     model_name = 'forest'
    #     ee = ExecutionEngine(name)
    #     ns_dict = {'ns_public': f'{name}',
    #                GlossaryCore.NS_WITNESS: f'{name}.{model_name}',
    #                GlossaryCore.NS_FUNCTIONS: f'{name}.{model_name}',
    #                'ns_forest': f'{name}.{model_name}',
    #                'ns_agriculture': f'{name}.{model_name}',
    #                'ns_invest': f'{name}.{model_name}'}
    #
    #     ee.ns_manager.add_ns_def(ns_dict)
    #
    #     mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline'
    #     builder = ee.factory.get_builder_from_module(model_name, mod_path)
    #
    #     ee.factory.set_builders_to_coupling_builder(builder)
    #
    #     ee.configure()
    #     ee.display_treeview_nodes()
    #
    #     self.year_start = GlossaryCore.YearStartDefault
    #     self.year_end = 2100
    #     years = np.arange(self.year_start, self.year_end + 1, 1)
    #     year_range = self.year_end - self.year_start + 1
    #
    #     '''
    #     Invests
    #     '''
    #     # 38.6 [G$] Investments in deforestation in 2021 [1]
    #     # 1.16 [Mha/year] of deforestation in 2050 => NZE would require reducing deforestation by two‐thirds by 2050 [wrt 2020?] [tab crop ref 1 p.92]
    #     # assume same trend until deforestation reaches 0 since eventually to preserve biodiversity and forest surface, forest will be either
    #     # protected or managed
    #     deforest_invest = np.concatenate((np.linspace(38.6, 12.83, 30), 12.83 * np.ones(year_range - 30)), axis=0) #np.concatenate((np.linspace(38.6, 12.83, 30), np.linspace(11.98, 0, 15), np.zeros((year_range - 30 - 15))), axis=0)
    #
    #     # 250 [Mha] of new forest to be planted by 2050 [crop tab ref 1 p 92]
    #     mw_invest = 1/3. * 0.027 * np.concatenate((np.linspace(5555., 8888., 20), 0. * np.ones(year_range - 20)), axis=0) # 3 years construction delay
    #     forest_invest = 0. * np.linspace(2, 10, year_range)
    #     '''
    #     End of invests
    #     '''
    #     self.forest_invest_df = pd.DataFrame(
    #         {GlossaryCore.Years: years, "forest_investment": forest_invest})
    #
    #
    #     self.mw_invest_df = pd.DataFrame(
    #         {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
    #
    #     # case over deforestation
    #     # deforest_invest = np.linspace(1000, 5000, year_range)
    #     self.deforest_invest_df = pd.DataFrame(
    #         {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: deforest_invest})
    #     inputs_dict = {f'{name}.{GlossaryCore.YearStart}': self.year_start,
    #                    f'{name}.{GlossaryCore.YearEnd}': self.year_end,
    #                    f'{name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}': self.deforest_invest_df,
    #                    f'{name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}': 8000,
    #                    f'{name}.{model_name}.{Forest.CO2_PER_HA}': self.CO2_per_ha,
    #                    f'{name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}': self.initial_emissions,
    #                    f'{name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}': self.forest_invest_df,
    #                    f'{name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}': self.reforestation_cost_per_ha,
    #                    f'{name}.{model_name}.managed_wood_initial_surface': 1.25 * 0.92,
    #                    f'{name}.{model_name}.managed_wood_invest_before_year_start': self.invest_before_year_start,
    #                    f'{name}.{model_name}.managed_wood_investment': self.mw_invest_df,
    #                    f'{name}.{model_name}.transport_cost': self.transport_df,
    #                    f'{name}.{model_name}.margin': self.margin,
    #                    f'{name}.{model_name}.initial_unmanaged_forest_surface': self.initial_unmanaged_forest_surface,
    #                    f'{name}.{model_name}.protected_forest_surface': self.initial_protected_forest_surface,
    #                    f'{name}.{model_name}.{GlossaryCore.CheckRangeBeforeRunBoolName}': False,
    #                    }
    #
    #     ee.load_study_from_input_dict(inputs_dict)
    #
    #     ee.execute()
    #
    #     disc = ee.dm.get_disciplines_with_name(
    #         f'{name}.{model_name}')[0]
    #     filter = disc.get_chart_filter_list()
    #     graph_list = disc.get_post_processing_list(filter)
    #
    #     for graph in graph_list:
    #         graph.to_plotly().show()

    def test_forest_discipline_low_deforestation(self):
        '''
        Check discipline setup and run
        '''

        name = 'Test'
        model_name = 'forest'
        ee = ExecutionEngine(name)
        ns_dict = {'ns_public': f'{name}',
                   GlossaryCore.NS_WITNESS: f'{name}.{model_name}',
                   GlossaryCore.NS_FUNCTIONS: f'{name}.{model_name}',
                   'ns_forest': f'{name}.{model_name}',
                   'ns_agriculture': f'{name}.{model_name}',
                   'ns_invest': f'{name}.{model_name}'}

        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline'
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2050
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        forest_invest = np.linspace(2, 10, year_range)
        self.forest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, "forest_investment": forest_invest})

        mw_invest = np.linspace(10, 10, year_range)
        self.mw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        deforest_invest = np.linspace(10, 5, year_range)
        # case over deforestation
        # deforest_invest = np.linspace(1000, 5000, year_range)
        self.deforest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: deforest_invest})
        inputs_dict = {f'{name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}': self.deforest_invest_df,
                       f'{name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}': 8000,
                       f'{name}.{model_name}.{Forest.CO2_PER_HA}': self.CO2_per_ha,
                       f'{name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}': self.initial_emissions,
                       f'{name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}': self.forest_invest_df,
                       f'{name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}': self.reforestation_cost_per_ha,
                       f'{name}.{model_name}.managed_wood_initial_surface': 1.25 * 0.92,
                       f'{name}.{model_name}.managed_wood_invest_before_year_start': self.invest_before_year_start,
                       f'{name}.{model_name}.managed_wood_investment': self.mw_invest_df,
                       f'{name}.{model_name}.transport_cost': self.transport_df,
                       f'{name}.{model_name}.margin': self.margin,
                       f'{name}.{model_name}.initial_unmanaged_forest_surface': self.initial_unmanaged_forest_surface,
                       f'{name}.{model_name}.protected_forest_surface': self.initial_protected_forest_surface,
                       }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(
            f'{name}.{model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)

    #         for graph in graph_list:
    #             graph.to_plotly().show()

    def test_forest_discipline_high_deforestation(self):
        '''
        Check discipline setup and run
        '''

        name = 'Test'
        model_name = 'forest'
        ee = ExecutionEngine(name)
        ns_dict = {'ns_public': f'{name}',
                   GlossaryCore.NS_WITNESS: f'{name}.{model_name}',
                   GlossaryCore.NS_FUNCTIONS: f'{name}.{model_name}',
                   'ns_forest': f'{name}.{model_name}',
                   'ns_agriculture': f'{name}.{model_name}',
                   'ns_invest': f'{name}.{model_name}'}

        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline'
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2050
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        forest_invest = np.linspace(2, 10, year_range)
        self.forest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, "forest_investment": forest_invest})

        mw_invest = np.linspace(300, 10, year_range)
        self.mw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        self.deforest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: np.linspace(2000., 1., len(years))})
        inputs_dict = {f'{name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}': self.deforest_invest_df,
                       f'{name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}': 8000,
                       f'{name}.{model_name}.{Forest.CO2_PER_HA}': self.CO2_per_ha,
                       f'{name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}': self.initial_emissions,
                       f'{name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}': self.forest_invest_df,
                       f'{name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}': self.reforestation_cost_per_ha,
                       f'{name}.{model_name}.managed_wood_initial_surface': 1.25 * 0.92,
                       f'{name}.{model_name}.managed_wood_invest_before_year_start': self.invest_before_year_start,
                       f'{name}.{model_name}.managed_wood_investment': self.mw_invest_df,
                       f'{name}.{model_name}.transport_cost': self.transport_df,
                       f'{name}.{model_name}.margin': self.margin,
                       f'{name}.{model_name}.initial_unmanaged_forest_surface': self.initial_unmanaged_forest_surface,
                       f'{name}.{model_name}.protected_forest_surface': self.initial_protected_forest_surface,
                       }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(
            f'{name}.{model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)

    #         for graph in graph_list:
    #             graph.to_plotly().show()

    def test_forest_discipline_high_lost_capital(self):
        '''
        Check discipline setup and run
        '''

        name = 'Test'
        model_name = 'forest'
        ee = ExecutionEngine(name)
        ns_dict = {'ns_public': f'{name}',
                   GlossaryCore.NS_WITNESS: f'{name}.{model_name}',
                   GlossaryCore.NS_FUNCTIONS: f'{name}.{model_name}',
                   'ns_forest': f'{name}.{model_name}',
                   'ns_agriculture': f'{name}.{model_name}',
                   'ns_invest': f'{name}.{model_name}'}

        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline'
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2080
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        construction_delay = 3
        self.invest_before_year_start = pd.DataFrame(
            {'past_years': np.arange(-construction_delay, 0),
             GlossaryCore.InvestmentsValue: np.array([1.135081] * construction_delay)})
        self.mw_initial_production = 23000  # TWh

        mw_invest = np.linspace(10, 10, year_range)
        self.mw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        transport = np.linspace(7.6, 7.6, year_range)
        self.transport_df = pd.DataFrame(
            {GlossaryCore.Years: years, "transport": transport})
        self.margin = pd.DataFrame(
            {GlossaryCore.Years: years, 'margin': np.ones(len(years)) * 110.0})
        self.forest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years,
             'forest_investment': np.linspace(1000., 1., len(years))})

        mw_invest = np.linspace(300, 10, year_range)
        self.mw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        self.deforest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: 40.})
        inputs_dict = {f'{name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}': self.deforest_invest_df,
                       f'{name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}': 8000,
                       f'{name}.{model_name}.{Forest.CO2_PER_HA}': self.CO2_per_ha,
                       f'{name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}': self.initial_emissions,
                       f'{name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}': self.forest_invest_df,
                       f'{name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}': self.reforestation_cost_per_ha,
                       f'{name}.{model_name}.managed_wood_initial_surface': 1.25 * 0.92,
                       f'{name}.{model_name}.managed_wood_invest_before_year_start': self.invest_before_year_start,
                       f'{name}.{model_name}.managed_wood_investment': self.mw_invest_df,
                       f'{name}.{model_name}.transport_cost': self.transport_df,
                       f'{name}.{model_name}.margin': self.margin,
                       f'{name}.{model_name}.initial_unmanaged_forest_surface': self.initial_unmanaged_forest_surface,
                       f'{name}.{model_name}.protected_forest_surface': self.initial_protected_forest_surface,
                       }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(
            f'{name}.{model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)

    #         for graph in graph_list:
    #             graph.to_plotly().show()

    def test_forest_discipline_low_lost_capital(self):
        '''
        Check discipline setup and run
        '''

        name = 'Test'
        model_name = 'forest'
        ee = ExecutionEngine(name)
        ns_dict = {'ns_public': f'{name}',
                   GlossaryCore.NS_WITNESS: f'{name}.{model_name}',
                   GlossaryCore.NS_FUNCTIONS: f'{name}.{model_name}',
                   'ns_forest': f'{name}.{model_name}',
                   'ns_agriculture': f'{name}.{model_name}',
                   'ns_invest': f'{name}.{model_name}'}

        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline'
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2080
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        construction_delay = 3
        self.invest_before_year_start = pd.DataFrame(
            {'past_years': np.arange(-construction_delay, 0),
             GlossaryCore.InvestmentsValue: np.array([1.135081] * construction_delay)})
        self.mw_initial_production = 23000  # TWh

        mw_invest = np.linspace(10, 10, year_range)
        self.mw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        transport = np.linspace(7.6, 7.6, year_range)
        self.transport_df = pd.DataFrame(
            {GlossaryCore.Years: years, "transport": transport})
        self.margin = pd.DataFrame(
            {GlossaryCore.Years: years, 'margin': np.ones(len(years)) * 110.0})
        self.forest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years,
             'forest_investment': np.linspace(100., 1., len(years))})

        mw_invest = np.linspace(300, 10, year_range)
        self.mw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        self.deforest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: 40.})
        inputs_dict = {f'{name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}': self.deforest_invest_df,
                       f'{name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}': 8000,
                       f'{name}.{model_name}.{Forest.CO2_PER_HA}': self.CO2_per_ha,
                       f'{name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}': self.initial_emissions,
                       f'{name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}': self.forest_invest_df,
                       f'{name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}': self.reforestation_cost_per_ha,
                       f'{name}.{model_name}.managed_wood_initial_surface': 1.25 * 0.92,
                       f'{name}.{model_name}.managed_wood_invest_before_year_start': self.invest_before_year_start,
                       f'{name}.{model_name}.managed_wood_investment': self.mw_invest_df,
                       f'{name}.{model_name}.transport_cost': self.transport_df,
                       f'{name}.{model_name}.margin': self.margin,
                       f'{name}.{model_name}.initial_unmanaged_forest_surface': self.initial_unmanaged_forest_surface,
                       f'{name}.{model_name}.protected_forest_surface': self.initial_protected_forest_surface,
                       }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(
            f'{name}.{model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)

    #         for graph in graph_list:
    #             graph.to_plotly().show()

    def test_forest_discipline_equal_invests(self):
        '''
        Check discipline setup and run
        '''

        name = 'Test'
        model_name = 'forest'
        ee = ExecutionEngine(name)
        ns_dict = {'ns_public': f'{name}',
                   GlossaryCore.NS_WITNESS: f'{name}.{model_name}',
                   GlossaryCore.NS_FUNCTIONS: f'{name}.{model_name}',
                   'ns_forest': f'{name}.{model_name}',
                   'ns_agriculture': f'{name}.{model_name}',
                   'ns_invest': f'{name}.{model_name}'}

        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline'
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        self.year_start =GlossaryCore.YearStartDefault
        self.year_end = 2080
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        construction_delay = 3
        self.invest_before_year_start = pd.DataFrame(
            {'past_years': np.arange(-construction_delay, 0),
             GlossaryCore.InvestmentsValue: np.array([1.135081] * construction_delay)})
        self.mw_initial_production = 23000  # TWh
        transport = np.linspace(7.6, 7.6, year_range)
        self.transport_df = pd.DataFrame(
            {GlossaryCore.Years: years, "transport": transport})
        self.margin = pd.DataFrame(
            {GlossaryCore.Years: years, 'margin': np.ones(len(years)) * 110.0})
        forest_invest = np.linspace(10, 50, year_range)
        self.forest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, "forest_investment": forest_invest})

        mw_invest = np.linspace(10, 10, year_range)
        self.mw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        deforest_invest = np.linspace(5.7971, 28.98550, year_range)
        # case over deforestation
        # deforest_invest = np.linspace(1000, 5000, year_range)
        self.deforest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: deforest_invest})

        mw_invest = np.linspace(300, 10, year_range)
        self.mw_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})

        inputs_dict = {f'{name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}': self.deforest_invest_df,
                       f'{name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}': 8000,
                       f'{name}.{model_name}.{Forest.CO2_PER_HA}': self.CO2_per_ha,
                       f'{name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}': self.initial_emissions,
                       f'{name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}': self.forest_invest_df,
                       f'{name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}': self.reforestation_cost_per_ha,
                       f'{name}.{model_name}.managed_wood_initial_surface': 1.25 * 0.92,
                       f'{name}.{model_name}.managed_wood_invest_before_year_start': self.invest_before_year_start,
                       f'{name}.{model_name}.managed_wood_investment': self.mw_invest_df,
                       f'{name}.{model_name}.transport_cost': self.transport_df,
                       f'{name}.{model_name}.margin': self.margin,
                       f'{name}.{model_name}.initial_unmanaged_forest_surface': self.initial_unmanaged_forest_surface,
                       f'{name}.{model_name}.protected_forest_surface': self.initial_protected_forest_surface,
                       }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(
            f'{name}.{model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
#         for graph in graph_list:
#             graph.to_plotly().show()
