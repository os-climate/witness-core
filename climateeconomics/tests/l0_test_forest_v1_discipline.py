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

import unittest

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.core.core_forest.forest_v1 import Forest
from climateeconomics.glossarycore import GlossaryCore


class ForestTestCase(unittest.TestCase):

    def setUp(self):
        """
        Initialize third data needed for testing
        """
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2055
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        deforestation_surface = np.array(np.linspace(10, 100, year_range))
        self.deforestation_surface_df = pd.DataFrame(
            {GlossaryCore.Years: years, "deforested_surface": deforestation_surface}
        )
        self.CO2_per_ha = 4000
        # Mha
        self.limit_deforestation_surface = 1000
        # GtCO2
        self.initial_emissions = 3.21
        forest_invest = np.linspace(2, 10, year_range)
        self.forest_invest_df = pd.DataFrame({GlossaryCore.Years: years, "forest_investment": forest_invest})
        self.reforestation_cost_per_ha = 3800

        self.param = {
            GlossaryCore.YearStart: self.year_start,
            GlossaryCore.YearEnd: self.year_end,
            GlossaryCore.TimeStep: self.time_step,
            Forest.DEFORESTATION_SURFACE: self.deforestation_surface_df,
            Forest.LIMIT_DEFORESTATION_SURFACE: self.limit_deforestation_surface,
            Forest.CO2_PER_HA: self.CO2_per_ha,
            Forest.INITIAL_CO2_EMISSIONS: self.initial_emissions,
            Forest.REFORESTATION_INVESTMENT: self.forest_invest_df,
            Forest.REFORESTATION_COST_PER_HA: self.reforestation_cost_per_ha,
            GlossaryCore.CheckRangeBeforeRunBoolName: False,
        }

    def test_forest_model(self):
        """
        Basique test of forest pyworld3
        Mainly check the overal run without value checks (will be done in another test)
        """

        forest = Forest(self.param)

        forest.compute(self.param)

    def test_forest_discipline(self):
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
            "ns_invest": f"{name}.{model_name}",
        }
        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = "climateeconomics.sos_wrapping.sos_wrapping_forest.forest_v1.forest_disc.ForestDiscipline"
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        inputs_dict = {
            f"{name}.{GlossaryCore.YearStart}": self.year_start,
            f"{name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{name}.{GlossaryCore.TimeStep}": 1,
            f"{name}.{model_name}.{Forest.LIMIT_DEFORESTATION_SURFACE}": self.limit_deforestation_surface,
            f"{name}.{model_name}.{Forest.DEFORESTATION_SURFACE}": self.deforestation_surface_df,
            f"{name}.{model_name}.{Forest.CO2_PER_HA}": self.CO2_per_ha,
            f"{name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}": self.initial_emissions,
            f"{name}.{model_name}.{Forest.REFORESTATION_INVESTMENT}": self.forest_invest_df,
            f"{name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}": self.reforestation_cost_per_ha,
        }

        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.dm.get_disciplines_with_name(f"{name}.{model_name}")[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)


#        for graph in graph_list:
#            graph.to_plotly().show()
