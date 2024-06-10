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

from climateeconomics.core.core_agriculture.agriculture import Agriculture


class AgricultureTestCase(unittest.TestCase):

    def setUp(self):
        """
        Initialize third data needed for testing
        """
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2055
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1

        population = np.array(np.linspace(7800, 7800, year_range))
        self.population_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.PopulationValue: population})
        self.population_df.index = years
        temperature = np.array(np.linspace(0.0, 0.0, year_range))
        self.temperature_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.TempAtmo: temperature})
        self.temperature_df.index = years

        self.default_kg_to_m2 = {
            "red meat": 360,
            "white meat": 16,
            "milk": 8.95,
            "eggs": 6.3,
            "rice and maize": 2.9,
            "potatoes": 0.88,
            "fruits and vegetables": 0.8,
        }
        self.default_kg_to_kcal = {
            "red meat": 2566,
            "white meat": 1860,
            "milk": 550,
            "eggs": 1500,
            "rice and maize": 1150,
            "potatoes": 670,
            "fruits and vegetables": 624,
        }
        red_meat_percentage = np.linspace(6, 1, year_range)
        white_meat_percentage = np.linspace(14, 5, year_range)
        self.red_meat_percentage = pd.DataFrame({GlossaryCore.Years: years, "red_meat_percentage": red_meat_percentage})
        self.white_meat_percentage = pd.DataFrame(
            {GlossaryCore.Years: years, "white_meat_percentage": white_meat_percentage}
        )

        self.diet_df = pd.DataFrame(
            {
                "red meat": [11.02],
                "white meat": [31.11],
                "milk": [79.27],
                "eggs": [9.68],
                "rice and maize": [97.76],
                "potatoes": [32.93],
                "fruits and vegetables": [217.62],
            }
        )
        self.other = np.array(np.linspace(0.102, 0.102, year_range))

        self.param = {
            GlossaryCore.YearStart: self.year_start,
            GlossaryCore.YearEnd: self.year_end,
            GlossaryCore.TimeStep: self.time_step,
            "diet_df": self.diet_df,
            "kg_to_kcal_dict": self.default_kg_to_kcal,
            GlossaryCore.PopulationDfValue: self.population_df,
            GlossaryCore.TemperatureDfValue: self.temperature_df,
            "kg_to_m2_dict": self.default_kg_to_m2,
            "red_meat_percentage": self.red_meat_percentage,
            "white_meat_percentage": self.white_meat_percentage,
            "other_use_agriculture": self.other,
            "param_a": -0.00833,
            "param_b": -0.04167,
            GlossaryCore.CheckRangeBeforeRunBoolName: False,
        }

    def test_agriculture_model(self):
        """
        Basique test of agriculture pyworld3
        Mainly check the overal run without value checks (will be done in another test)
        """

        agriculture = Agriculture(self.param)
        agriculture.apply_percentage(self.param)
        agriculture.compute(self.population_df, self.temperature_df)

    def test_agriculture_discipline(self):
        """
        Check discipline setup and run
        """

        name = "Test"
        model_name = "agriculture"
        ee = ExecutionEngine(name)
        ns_dict = {
            "ns_public": f"{name}",
            GlossaryCore.NS_WITNESS: f"{name}.{model_name}",
            GlossaryCore.NS_FUNCTIONS: f"{name}.{model_name}",
            "ns_agriculture": f"{name}.{model_name}",
        }
        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = (
            "climateeconomics.sos_wrapping.sos_wrapping_agriculture.agriculture.agriculture_disc.AgricultureDiscipline"
        )
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        inputs_dict = {
            f"{name}.{GlossaryCore.YearStart}": self.year_start,
            f"{name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{name}.{GlossaryCore.TimeStep}": 1,
            f"{name}.{model_name}.{Agriculture.DIET_DF}": self.diet_df,
            f"{name}.{model_name}.{Agriculture.KG_TO_KCAL_DICT}": self.default_kg_to_kcal,
            f"{name}.{model_name}.{Agriculture.KG_TO_M2_DICT}": self.default_kg_to_m2,
            f"{name}.{model_name}.{Agriculture.POPULATION_DF}": self.population_df,
            f"{name}.{model_name}.red_meat_percentage": self.red_meat_percentage,
            f"{name}.{model_name}.white_meat_percentage": self.white_meat_percentage,
            f"{name}.{model_name}.{Agriculture.OTHER_USE_AGRICULTURE}": self.other,
            f"{name}.{model_name}.{GlossaryCore.TemperatureDfValue}": self.temperature_df,
        }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(f"{name}.{model_name}")[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        # for graph in graph_list:
        #    graph.to_plotly().show()
