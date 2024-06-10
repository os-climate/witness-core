"""
Copyright 2022 Airbus SAS
Modifications on 27/11/2023 Copyright 2023 Capgemini

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

import random as rd
import unittest

import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.glossarycore import GlossaryCore


class TestSoSDiscipline(unittest.TestCase):

    def setUp(self):
        self.name = "Test"
        self.ee = ExecutionEngine(self.name)
        self.factory = self.ee.factory

    def test_01_execute_process(self):
        model_name = "CopperModel"
        ns_dict = {"ns_public": f"{self.name}"}

        self.ee.ns_manager.add_ns_def(ns_dict)
        mod_path = "climateeconomics.sos_wrapping.sos_wrapping_resources.sos_wrapping_copper_resource_v0.copper_disc.CopperDisc"
        builder = self.factory.get_builder_from_module(model_name, mod_path)

        self.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        year = GlossaryCore.YearStartDefault
        year_end = GlossaryCore.YearEndDefault + 1

        copper_demand = pd.DataFrame([(year, rd.gauss(26, 0.5), "million_tonnes")], columns=["Year", "Demand", "unit"])
        extraction = [26]

        year += 1

        while year < year_end:
            ref_series = pd.Series(
                {
                    "Year": year,
                    "Demand": rd.gauss(26, 0.5) * (1.056467) ** (year - GlossaryCore.YearStartDefault),
                    "unit": "million_tonnes",
                }
            )
            copper_demand = pd.concat([copper_demand, pd.DataFrame([ref_series])], ignore_index=True)
            extraction += [26 * (1.056467) ** (year - GlossaryCore.YearStartDefault)]
            year += 1

        values_dict = {
            "Test.CopperModel.copper_demand": copper_demand,
            "Test.CopperModel.annual_extraction": extraction,
        }
        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        copper_model = self.ee.dm.get_disciplines_with_name("Test.CopperModel")[0]
        filters = copper_model.get_chart_filter_list()
        graph_list = copper_model.get_post_processing_list(filters)


#         for graph in graph_list:
#             graph.to_plotly().show()


if __name__ == "__main__":
    unittest.main()
