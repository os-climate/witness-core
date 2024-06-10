"""
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
"""

from climateeconomics.glossarycore import GlossaryCore

"""
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
"""
import unittest

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.core.core_forest.forest_v2 import Forest


class ForestTestCase(unittest.TestCase):

    def setUp(self):
        """
        Initialize third data needed for testing
        """
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2050
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        self.CO2_per_ha = 4000
        # GtCO2
        self.initial_emissions = -7.6
        forest_invest = np.linspace(45, 50, year_range)
        self.forest_invest_df = pd.DataFrame({GlossaryCore.Years: years, "forest_investment": forest_invest})
        self.reforestation_cost_per_ha = 13800

        construction_delay = 3

        self.invest_before_year_start = pd.DataFrame(
            {
                "past_years": np.arange(-construction_delay, 0),
                GlossaryCore.InvestmentsValue: np.array([1.135081] * construction_delay),
            }
        )

        mw_invest = np.linspace(10, 10, year_range)
        self.mw_invest_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        transport = np.linspace(7.6, 7.6, year_range)
        self.transport_df = pd.DataFrame({GlossaryCore.Years: years, "transport": transport})
        self.margin = pd.DataFrame({GlossaryCore.Years: years, "margin": np.ones(len(years)) * 110.0})
        self.initial_protected_forest_surface = 4 * 0.21
        self.initial_unmanaged_forest_surface = 4 - 1.25 - self.initial_protected_forest_surface
        deforest_invest = np.linspace(10, 5, year_range)
        # case over deforestation
        # deforest_invest = np.linspace(1000, 5000, year_range)
        self.deforest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: deforest_invest}
        )

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
    #     self.time_step = 1
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
    #                    f'{name}.{GlossaryCore.TimeStep}': 1,
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
        """
        Check discipline setup and run
        """

        name = "Test"
        model_name = "forest"
        ee = ExecutionEngine(name)
        ns_dict = {
            "ns_public": f"{name}",
            GlossaryCore.NS_WITNESS: f"{name}.{model_name}",
            GlossaryCore.NS_FUNCTIONS: f"{name}.{model_name}",
            "ns_forest": f"{name}.{model_name}",
            "ns_agriculture": f"{name}.{model_name}",
            "ns_invest": f"{name}.{model_name}",
        }

        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = "climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline"
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2050
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        forest_invest = np.linspace(2, 10, year_range)
        self.forest_invest_df = pd.DataFrame({GlossaryCore.Years: years, "forest_investment": forest_invest})

        mw_invest = np.linspace(10, 10, year_range)
        self.mw_invest_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        deforest_invest = np.linspace(10, 5, year_range)
        # case over deforestation
        # deforest_invest = np.linspace(1000, 5000, year_range)
        self.deforest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: deforest_invest}
        )
        inputs_dict = {
            f"{name}.{GlossaryCore.YearStart}": self.year_start,
            f"{name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{name}.{GlossaryCore.TimeStep}": 1,
            f"{name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}": self.deforest_invest_df,
            f"{name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}": 8000,
            f"{name}.{model_name}.{Forest.CO2_PER_HA}": self.CO2_per_ha,
            f"{name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}": self.initial_emissions,
            f"{name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}": self.forest_invest_df,
            f"{name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}": self.reforestation_cost_per_ha,
            f"{name}.{model_name}.managed_wood_initial_surface": 1.25 * 0.92,
            f"{name}.{model_name}.managed_wood_invest_before_year_start": self.invest_before_year_start,
            f"{name}.{model_name}.managed_wood_investment": self.mw_invest_df,
            f"{name}.{model_name}.transport_cost": self.transport_df,
            f"{name}.{model_name}.margin": self.margin,
            f"{name}.{model_name}.initial_unmanaged_forest_surface": self.initial_unmanaged_forest_surface,
            f"{name}.{model_name}.protected_forest_surface": self.initial_protected_forest_surface,
        }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(f"{name}.{model_name}")[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)

    #         for graph in graph_list:
    #             graph.to_plotly().show()

    def test_forest_discipline_high_deforestation(self):
        """
        Check discipline setup and run
        """

        name = "Test"
        model_name = "forest"
        ee = ExecutionEngine(name)
        ns_dict = {
            "ns_public": f"{name}",
            GlossaryCore.NS_WITNESS: f"{name}.{model_name}",
            GlossaryCore.NS_FUNCTIONS: f"{name}.{model_name}",
            "ns_forest": f"{name}.{model_name}",
            "ns_agriculture": f"{name}.{model_name}",
            "ns_invest": f"{name}.{model_name}",
        }

        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = "climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline"
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2050
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        forest_invest = np.linspace(2, 10, year_range)
        self.forest_invest_df = pd.DataFrame({GlossaryCore.Years: years, "forest_investment": forest_invest})

        mw_invest = np.linspace(300, 10, year_range)
        self.mw_invest_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        self.deforest_invest_df = pd.DataFrame(
            {
                GlossaryCore.Years: years,
                GlossaryCore.InvestmentsValue: np.array(
                    [
                        2000.00,
                        2000.00,
                        2000.00,
                        2000.00,
                        2000.00,
                        2000.00,
                        2000.00,
                        2000.00,
                        2000.00,
                        2000.00,
                        2000.00,
                        2000.00,
                        2000.00,
                        2000.00,
                        2000.00,
                        2000.00,
                        2000.00,
                        2000.00,
                        0.00,
                        0.00,
                        0.00,
                        0.00,
                        0.00,
                        0.00,
                        0.00,
                        0.00,
                        0.00,
                        0.00,
                        0.00,
                        0.00,
                        0.00,
                    ]
                ),
            }
        )
        inputs_dict = {
            f"{name}.{GlossaryCore.YearStart}": self.year_start,
            f"{name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{name}.{GlossaryCore.TimeStep}": 1,
            f"{name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}": self.deforest_invest_df,
            f"{name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}": 8000,
            f"{name}.{model_name}.{Forest.CO2_PER_HA}": self.CO2_per_ha,
            f"{name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}": self.initial_emissions,
            f"{name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}": self.forest_invest_df,
            f"{name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}": self.reforestation_cost_per_ha,
            f"{name}.{model_name}.managed_wood_initial_surface": 1.25 * 0.92,
            f"{name}.{model_name}.managed_wood_invest_before_year_start": self.invest_before_year_start,
            f"{name}.{model_name}.managed_wood_investment": self.mw_invest_df,
            f"{name}.{model_name}.transport_cost": self.transport_df,
            f"{name}.{model_name}.margin": self.margin,
            f"{name}.{model_name}.initial_unmanaged_forest_surface": self.initial_unmanaged_forest_surface,
            f"{name}.{model_name}.protected_forest_surface": self.initial_protected_forest_surface,
        }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(f"{name}.{model_name}")[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)

    #         for graph in graph_list:
    #             graph.to_plotly().show()

    def test_forest_discipline_high_lost_capital(self):
        """
        Check discipline setup and run
        """

        name = "Test"
        model_name = "forest"
        ee = ExecutionEngine(name)
        ns_dict = {
            "ns_public": f"{name}",
            GlossaryCore.NS_WITNESS: f"{name}.{model_name}",
            GlossaryCore.NS_FUNCTIONS: f"{name}.{model_name}",
            "ns_forest": f"{name}.{model_name}",
            "ns_agriculture": f"{name}.{model_name}",
            "ns_invest": f"{name}.{model_name}",
        }

        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = "climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline"
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2080
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        construction_delay = 3
        self.invest_before_year_start = pd.DataFrame(
            {
                "past_years": np.arange(-construction_delay, 0),
                GlossaryCore.InvestmentsValue: np.array([1.135081] * construction_delay),
            }
        )
        self.mw_initial_production = 23000  # TWh

        mw_invest = np.linspace(10, 10, year_range)
        self.mw_invest_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        transport = np.linspace(7.6, 7.6, year_range)
        self.transport_df = pd.DataFrame({GlossaryCore.Years: years, "transport": transport})
        self.margin = pd.DataFrame({GlossaryCore.Years: years, "margin": np.ones(len(years)) * 110.0})
        self.forest_invest_df = pd.DataFrame(
            {
                GlossaryCore.Years: years,
                "forest_investment": np.array(
                    [
                        262.3080913163442,
                        469.0288051740345,
                        646.7923512783858,
                        797.3169880002974,
                        922.3209737106685,
                        1023.5225667803987,
                        1102.6400255803876,
                        1161.3916084815337,
                        1201.4955738547371,
                        1224.670180070897,
                        1232.6336855009129,
                        1227.1043485156838,
                        1209.8004274861094,
                        1182.4401807830886,
                        1146.7418667775214,
                        1104.4237438403065,
                        1057.2040703423438,
                        1006.5964839949823,
                        953.2961398713705,
                        897.7935723851078,
                        840.579315949792,
                        782.1439049790221,
                        722.9778738863969,
                        663.5717570855144,
                        604.416088989974,
                        546.001404013374,
                        488.818236569313,
                        433.35712107138966,
                        380.1085919332026,
                        329.5631835683504,
                        282.2114303904319,
                        238.5438668130455,
                        199.05102724978983,
                        164.0929896380927,
                        133.50800601069787,
                        107.00387192417816,
                        84.28838293510618,
                        65.06933460005494,
                        49.054522475597174,
                        35.95174211830572,
                        25.468789084753404,
                        17.31345893151294,
                        11.193547215157256,
                        6.81684949225909,
                        3.8911613193912378,
                        2.1242782531265565,
                        1.2239958500378234,
                        0.8981096666978611,
                        0.854415259679471,
                        0.849695588213618,
                        0.836683222163885,
                        0.8170981340520096,
                        0.7926602963997317,
                        0.7650896817287889,
                        0.7361062625609198,
                        0.707430011417863,
                        0.680780900821357,
                        0.6578789032931407,
                        0.6404439913549522,
                        0.6301961375285303,
                        0.6288553143356134,
                    ]
                ),
            }
        )

        mw_invest = np.linspace(300, 10, year_range)
        self.mw_invest_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        self.deforest_invest_df = pd.DataFrame(
            {
                GlossaryCore.Years: years,
                GlossaryCore.InvestmentsValue: np.array(
                    [
                        40.62218956750874,
                        40.56970502780573,
                        40.53622422620158,
                        40.52075010891646,
                        40.52228562217051,
                        40.53983371218389,
                        40.57239732517679,
                        40.618979407369324,
                        40.678582904981674,
                        40.750210764233984,
                        40.83286593134644,
                        40.925551352539166,
                        41.02726997403234,
                        41.13702474204612,
                        41.253818602800656,
                        41.37665450251611,
                        41.50453538741265,
                        41.636519367006834,
                        41.771885204000874,
                        41.90996682439344,
                        42.050098154183154,
                        42.19161311936864,
                        42.33384564594855,
                        42.47612965992152,
                        42.61779908728619,
                        42.758187854041175,
                        42.896629886185124,
                        43.032459109716704,
                        43.16500945063452,
                        43.2936148349372,
                        43.41760918862341,
                        43.53632643769176,
                        43.649100508140904,
                        43.75535489927214,
                        43.85487140359732,
                        43.94752138693098,
                        44.03317621508764,
                        44.111707253881846,
                        44.1829858691281,
                        44.24688342664097,
                        44.30327129223493,
                        44.35202083172455,
                        44.39300341092434,
                        44.426090395648814,
                        44.45115315171254,
                        44.46806304492999,
                        44.47669144111572,
                        44.47690970608428,
                        44.46858920565015,
                        44.44960686136805,
                        44.40986181775338,
                        44.33725877506167,
                        44.2197024335485,
                        44.04509749346942,
                        43.80134865507997,
                        43.47636061863573,
                        43.058038084392244,
                        42.534285752605086,
                        41.893008323529784,
                        41.12211049742193,
                        40.20949697453705,
                    ]
                ),
            }
        )
        inputs_dict = {
            f"{name}.{GlossaryCore.YearStart}": self.year_start,
            f"{name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{name}.{GlossaryCore.TimeStep}": 1,
            f"{name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}": self.deforest_invest_df,
            f"{name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}": 8000,
            f"{name}.{model_name}.{Forest.CO2_PER_HA}": self.CO2_per_ha,
            f"{name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}": self.initial_emissions,
            f"{name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}": self.forest_invest_df,
            f"{name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}": self.reforestation_cost_per_ha,
            f"{name}.{model_name}.managed_wood_initial_surface": 1.25 * 0.92,
            f"{name}.{model_name}.managed_wood_invest_before_year_start": self.invest_before_year_start,
            f"{name}.{model_name}.managed_wood_investment": self.mw_invest_df,
            f"{name}.{model_name}.transport_cost": self.transport_df,
            f"{name}.{model_name}.margin": self.margin,
            f"{name}.{model_name}.initial_unmanaged_forest_surface": self.initial_unmanaged_forest_surface,
            f"{name}.{model_name}.protected_forest_surface": self.initial_protected_forest_surface,
        }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(f"{name}.{model_name}")[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)

    #         for graph in graph_list:
    #             graph.to_plotly().show()

    def test_forest_discipline_low_lost_capital(self):
        """
        Check discipline setup and run
        """

        name = "Test"
        model_name = "forest"
        ee = ExecutionEngine(name)
        ns_dict = {
            "ns_public": f"{name}",
            GlossaryCore.NS_WITNESS: f"{name}.{model_name}",
            GlossaryCore.NS_FUNCTIONS: f"{name}.{model_name}",
            "ns_forest": f"{name}.{model_name}",
            "ns_agriculture": f"{name}.{model_name}",
            "ns_invest": f"{name}.{model_name}",
        }

        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = "climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline"
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2080
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        construction_delay = 3
        self.invest_before_year_start = pd.DataFrame(
            {
                "past_years": np.arange(-construction_delay, 0),
                GlossaryCore.InvestmentsValue: np.array([1.135081] * construction_delay),
            }
        )
        self.mw_initial_production = 23000  # TWh

        mw_invest = np.linspace(10, 10, year_range)
        self.mw_invest_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        transport = np.linspace(7.6, 7.6, year_range)
        self.transport_df = pd.DataFrame({GlossaryCore.Years: years, "transport": transport})
        self.margin = pd.DataFrame({GlossaryCore.Years: years, "margin": np.ones(len(years)) * 110.0})
        self.forest_invest_df = pd.DataFrame(
            {
                GlossaryCore.Years: years,
                "forest_investment": np.array(
                    [
                        26.3080913163442,
                        46.0288051740345,
                        64.7923512783858,
                        79.3169880002974,
                        92.3209737106685,
                        102.5225667803987,
                        110.6400255803876,
                        116.3916084815337,
                        120.4955738547371,
                        122.670180070897,
                        123.6336855009129,
                        122.1043485156838,
                        120.8004274861094,
                        118.4401807830886,
                        114.7418667775214,
                        110.4237438403065,
                        105.2040703423438,
                        100.5964839949823,
                        95.2961398713705,
                        89.7935723851078,
                        84.579315949792,
                        78.1439049790221,
                        72.9778738863969,
                        66.5717570855144,
                        60.416088989974,
                        54.001404013374,
                        48.818236569313,
                        43.35712107138966,
                        38.1085919332026,
                        32.5631835683504,
                        28.2114303904319,
                        23.5438668130455,
                        19.05102724978983,
                        16.0929896380927,
                        13.50800601069787,
                        10.00387192417816,
                        8.28838293510618,
                        6.06933460005494,
                        4.054522475597174,
                        3.95174211830572,
                        2.468789084753404,
                        1.31345893151294,
                        1.193547215157256,
                        0.81684949225909,
                        0.8911613193912378,
                        0.1242782531265565,
                        0.2239958500378234,
                        0.8981096666978611,
                        0.854415259679471,
                        0.849695588213618,
                        0.836683222163885,
                        0.8170981340520096,
                        0.7926602963997317,
                        0.7650896817287889,
                        0.7361062625609198,
                        0.707430011417863,
                        0.680780900821357,
                        0.6578789032931407,
                        0.6404439913549522,
                        0.6301961375285303,
                        0.6288553143356134,
                    ]
                ),
            }
        )

        mw_invest = np.linspace(300, 10, year_range)
        self.mw_invest_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        self.deforest_invest_df = pd.DataFrame(
            {
                GlossaryCore.Years: years,
                GlossaryCore.InvestmentsValue: np.array(
                    [
                        40.62218956750874,
                        40.56970502780573,
                        40.53622422620158,
                        40.52075010891646,
                        40.52228562217051,
                        40.53983371218389,
                        40.57239732517679,
                        40.618979407369324,
                        40.678582904981674,
                        40.750210764233984,
                        40.83286593134644,
                        40.925551352539166,
                        41.02726997403234,
                        41.13702474204612,
                        41.253818602800656,
                        41.37665450251611,
                        41.50453538741265,
                        41.636519367006834,
                        41.771885204000874,
                        41.90996682439344,
                        42.050098154183154,
                        42.19161311936864,
                        42.33384564594855,
                        42.47612965992152,
                        42.61779908728619,
                        42.758187854041175,
                        42.896629886185124,
                        43.032459109716704,
                        43.16500945063452,
                        43.2936148349372,
                        43.41760918862341,
                        43.53632643769176,
                        43.649100508140904,
                        43.75535489927214,
                        43.85487140359732,
                        43.94752138693098,
                        44.03317621508764,
                        44.111707253881846,
                        44.1829858691281,
                        44.24688342664097,
                        44.30327129223493,
                        44.35202083172455,
                        44.39300341092434,
                        44.426090395648814,
                        44.45115315171254,
                        44.46806304492999,
                        44.47669144111572,
                        44.47690970608428,
                        44.46858920565015,
                        44.44960686136805,
                        44.40986181775338,
                        44.33725877506167,
                        44.2197024335485,
                        44.04509749346942,
                        43.80134865507997,
                        43.47636061863573,
                        43.058038084392244,
                        42.534285752605086,
                        41.893008323529784,
                        41.12211049742193,
                        40.20949697453705,
                    ]
                ),
            }
        )
        inputs_dict = {
            f"{name}.{GlossaryCore.YearStart}": self.year_start,
            f"{name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{name}.{GlossaryCore.TimeStep}": 1,
            f"{name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}": self.deforest_invest_df,
            f"{name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}": 8000,
            f"{name}.{model_name}.{Forest.CO2_PER_HA}": self.CO2_per_ha,
            f"{name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}": self.initial_emissions,
            f"{name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}": self.forest_invest_df,
            f"{name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}": self.reforestation_cost_per_ha,
            f"{name}.{model_name}.managed_wood_initial_surface": 1.25 * 0.92,
            f"{name}.{model_name}.managed_wood_invest_before_year_start": self.invest_before_year_start,
            f"{name}.{model_name}.managed_wood_investment": self.mw_invest_df,
            f"{name}.{model_name}.transport_cost": self.transport_df,
            f"{name}.{model_name}.margin": self.margin,
            f"{name}.{model_name}.initial_unmanaged_forest_surface": self.initial_unmanaged_forest_surface,
            f"{name}.{model_name}.protected_forest_surface": self.initial_protected_forest_surface,
        }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(f"{name}.{model_name}")[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)

    #         for graph in graph_list:
    #             graph.to_plotly().show()

    def test_forest_discipline_equal_invests(self):
        """
        Check discipline setup and run
        """

        name = "Test"
        model_name = "forest"
        ee = ExecutionEngine(name)
        ns_dict = {
            "ns_public": f"{name}",
            GlossaryCore.NS_WITNESS: f"{name}.{model_name}",
            GlossaryCore.NS_FUNCTIONS: f"{name}.{model_name}",
            "ns_forest": f"{name}.{model_name}",
            "ns_agriculture": f"{name}.{model_name}",
            "ns_invest": f"{name}.{model_name}",
        }

        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = "climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline"
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2080
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        construction_delay = 3
        self.invest_before_year_start = pd.DataFrame(
            {
                "past_years": np.arange(-construction_delay, 0),
                GlossaryCore.InvestmentsValue: np.array([1.135081] * construction_delay),
            }
        )
        self.mw_initial_production = 23000  # TWh
        transport = np.linspace(7.6, 7.6, year_range)
        self.transport_df = pd.DataFrame({GlossaryCore.Years: years, "transport": transport})
        self.margin = pd.DataFrame({GlossaryCore.Years: years, "margin": np.ones(len(years)) * 110.0})
        forest_invest = np.linspace(10, 50, year_range)
        self.forest_invest_df = pd.DataFrame({GlossaryCore.Years: years, "forest_investment": forest_invest})

        mw_invest = np.linspace(10, 10, year_range)
        self.mw_invest_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})
        deforest_invest = np.linspace(5.7971, 28.98550, year_range)
        # case over deforestation
        # deforest_invest = np.linspace(1000, 5000, year_range)
        self.deforest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: deforest_invest}
        )

        mw_invest = np.linspace(300, 10, year_range)
        self.mw_invest_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: mw_invest})

        inputs_dict = {
            f"{name}.{GlossaryCore.YearStart}": self.year_start,
            f"{name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{name}.{GlossaryCore.TimeStep}": 1,
            f"{name}.{model_name}.{Forest.DEFORESTATION_INVESTMENT}": self.deforest_invest_df,
            f"{name}.{model_name}.{Forest.DEFORESTATION_COST_PER_HA}": 8000,
            f"{name}.{model_name}.{Forest.CO2_PER_HA}": self.CO2_per_ha,
            f"{name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}": self.initial_emissions,
            f"{name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}": self.forest_invest_df,
            f"{name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}": self.reforestation_cost_per_ha,
            f"{name}.{model_name}.managed_wood_initial_surface": 1.25 * 0.92,
            f"{name}.{model_name}.managed_wood_invest_before_year_start": self.invest_before_year_start,
            f"{name}.{model_name}.managed_wood_investment": self.mw_invest_df,
            f"{name}.{model_name}.transport_cost": self.transport_df,
            f"{name}.{model_name}.margin": self.margin,
            f"{name}.{model_name}.initial_unmanaged_forest_surface": self.initial_unmanaged_forest_surface,
            f"{name}.{model_name}.protected_forest_surface": self.initial_protected_forest_surface,
        }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(f"{name}.{model_name}")[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)


#         for graph in graph_list:
#             graph.to_plotly().show()
