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
from os.path import dirname, join

from pandas import read_csv
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.glossarycore import GlossaryCore


class AllResourceModelTestCase(unittest.TestCase):

    def setUp(self):
        """
        Initialize third data needed for testing
        """
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = GlossaryCore.YearEndDefault

        data_dir = join(dirname(__file__), "data")

        self.oil_production_df = read_csv(join(data_dir, "oil.predictible_production.csv"))
        self.oil_production_df.set_index(GlossaryCore.Years, inplace=True)
        self.gas_production_df = read_csv(join(data_dir, "gas.predictible_production.csv"))
        self.coal_production_df = read_csv(join(data_dir, "coal.predictible_production.csv"))
        self.uranium_production_df = read_csv(join(data_dir, "uranium.predictible_production.csv"))
        self.copper_production_df = read_csv(join(data_dir, "copper.predictible_production.csv"))
        self.platinum_production_df = read_csv(join(data_dir, "platinum.predictible_production.csv"))
        self.oil_stock_df = read_csv(join(data_dir, "oil.stock.csv"))
        self.gas_stock_df = read_csv(join(data_dir, "gas.stock.csv"))
        self.uranium_stock_df = read_csv(join(data_dir, "uranium.stock.csv"))
        self.copper_stock_df = read_csv(join(data_dir, "copper.stock.csv"))
        self.coal_stock_df = read_csv(join(data_dir, "coal.stock.csv"))
        self.platinum_stock_df = read_csv(join(data_dir, "platinum.stock.csv"))
        self.oil_price_df = read_csv(join(data_dir, "oil.price.csv"))
        self.gas_price_df = read_csv(join(data_dir, "gas.price.csv"))
        self.coal_price_df = read_csv(join(data_dir, "coal.price.csv"))
        self.uranium_price_df = read_csv(join(data_dir, "uranium.price.csv"))
        self.copper_price_df = read_csv(join(data_dir, "copper.price.csv"))
        self.platinum_price_df = read_csv(join(data_dir, "platinum.price.csv"))
        self.oil_use_df = read_csv(join(data_dir, "oil.use.csv"))
        self.gas_use_df = read_csv(join(data_dir, "gas.use.csv"))
        self.coal_use_df = read_csv(join(data_dir, "coal.use.csv"))
        self.uranium_use_df = read_csv(join(data_dir, "uranium.use.csv"))
        self.copper_use_df = read_csv(join(data_dir, "copper.use.csv"))
        self.platinum_use_df = read_csv(join(data_dir, "platinum.use.csv"))
        self.oil_recycled_production_df = read_csv(join(data_dir, "oil.recycled_production.csv"))
        self.gas_recycled_production_df = read_csv(join(data_dir, "gas.recycled_production.csv"))
        self.coal_recycled_production_df = read_csv(join(data_dir, "coal.recycled_production.csv"))
        self.uranium_recycled_production_df = read_csv(join(data_dir, "uranium.recycled_production.csv"))
        self.copper_recycled_production_df = read_csv(join(data_dir, "copper.recycled_production.csv"))
        self.platinum_recycled_production_df = read_csv(join(data_dir, "platinum.recycled_production.csv"))
        self.non_modeled_resource_df = read_csv(join(data_dir, "resource_data_price.csv"))
        self.all_demand = read_csv(join(data_dir, "all_demand_from_energy_mix.csv"))

        self.resource_list = [
            "natural_gas_resource",
            "uranium_resource",
            "coal_resource",
            "oil_resource",
            "copper_resource",
            "platinum_resource",
        ]

        self.param = {
            "All_Demand": self.all_demand,
            GlossaryCore.YearStart: self.year_start,
            GlossaryCore.YearEnd: self.year_end,
            "resource_list": self.resource_list,
            "oil_resource.predictible_production": self.oil_production_df,
            "natural_gas_resource.predictible_production": self.gas_production_df,
            "uranium_resource.predictible_production": self.uranium_production_df,
            "coal_resource.predictible_production": self.coal_production_df,
            "copper_resource.predictible_production": self.copper_production_df,
            "platinum_resource.predictible_production": self.platinum_production_df,
            "oil_resource.use_stock": self.oil_use_df,
            "natural_gas_resource.use_stock": self.gas_use_df,
            "uranium_resource.use_stock": self.uranium_use_df,
            "coal_resource.use_stock": self.coal_use_df,
            "copper_resource.use_stock": self.copper_use_df,
            "platinum_resource.use_stock": self.platinum_use_df,
            "oil_resource.resource_stock": self.oil_stock_df,
            "natural_gas_resource.resource_stock": self.gas_stock_df,
            "uranium_resource.resource_stock": self.uranium_stock_df,
            "coal_resource.resource_stock": self.coal_stock_df,
            "copper_resource.resource_stock": self.copper_stock_df,
            "platinum_resource.resource_stock": self.platinum_stock_df,
            "oil_resource.resource_price": self.oil_price_df,
            "natural_gas_resource.resource_price": self.gas_price_df,
            "coal_resource.resource_price": self.coal_price_df,
            "uranium_resource.resource_price": self.uranium_price_df,
            "copper_resource.resource_price": self.copper_price_df,
            "platinum_resource.resource_price": self.platinum_price_df,
            "non_modeled_resource_price": self.non_modeled_resource_df,
            "oil_resource.recycled_production": self.oil_recycled_production_df,
            "natural_gas_resource.recycled_production": self.gas_recycled_production_df,
            "uranium_resource.recycled_production": self.uranium_recycled_production_df,
            "coal_resource.recycled_production": self.coal_recycled_production_df,
            "copper_resource.recycled_production": self.copper_recycled_production_df,
            "platinum_resource.recycled_production": self.platinum_recycled_production_df,
        }

    def test_All_resource_discipline(self):
        """
        Check discipline setup and run
        """
        name = "Test"
        model_name = "All_resource"
        ee = ExecutionEngine(name)
        ns_dict = {
            "ns_public": f"{name}",
            "ns_coal_resource": f"{name}.{model_name}",
            "ns_oil_resource": f"{name}.{model_name}",
            "ns_natural_gas_resource": f"{name}.{model_name}",
            "ns_uranium_resource": f"{name}.{model_name}",
            "ns_copper_resource": f"{name}.{model_name}",
            "ns_platinum_resource": f"{name}.{model_name}",
            "ns_resource": f"{name}.{model_name}",
        }
        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = "climateeconomics.core.core_resources.resource_mix.resource_mix_disc.ResourceMixDiscipline"
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()
        inputs_dict = {
            f"{name}.{GlossaryCore.YearStart}": self.year_start,
            f"{name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{name}.{model_name}.resources_demand": self.all_demand,
            f"{name}.{model_name}.resources_demand_woratio": self.all_demand,
            f"{name}.{model_name}.oil_resource.predictable_production": self.oil_production_df,
            f"{name}.{model_name}.oil_resource.resource_stock": self.oil_stock_df,
            f"{name}.{model_name}.oil_resource.resource_price": self.oil_price_df,
            f"{name}.{model_name}.oil_resource.use_stock": self.oil_use_df,
            f"{name}.{model_name}.oil_resource.recycled_production": self.oil_recycled_production_df,
            f"{name}.{model_name}.natural_gas_resource.predictable_production": self.gas_production_df,
            f"{name}.{model_name}.natural_gas_resource.resource_stock": self.gas_stock_df,
            f"{name}.{model_name}.natural_gas_resource.resource_price": self.gas_price_df,
            f"{name}.{model_name}.natural_gas_resource.use_stock": self.gas_use_df,
            f"{name}.{model_name}.natural_gas_resource.recycled_production": self.gas_recycled_production_df,
            f"{name}.{model_name}.coal_resource.predictable_production": self.coal_production_df,
            f"{name}.{model_name}.coal_resource.resource_stock": self.coal_stock_df,
            f"{name}.{model_name}.coal_resource.resource_price": self.coal_price_df,
            f"{name}.{model_name}.coal_resource.use_stock": self.coal_use_df,
            f"{name}.{model_name}.coal_resource.recycled_production": self.coal_recycled_production_df,
            f"{name}.{model_name}.uranium_resource.predictable_production": self.uranium_production_df,
            f"{name}.{model_name}.uranium_resource.resource_stock": self.uranium_stock_df,
            f"{name}.{model_name}.uranium_resource.resource_price": self.uranium_price_df,
            f"{name}.{model_name}.uranium_resource.use_stock": self.uranium_use_df,
            f"{name}.{model_name}.uranium_resource.recycled_production": self.uranium_recycled_production_df,
            f"{name}.{model_name}.copper_resource.predictable_production": self.copper_production_df,
            f"{name}.{model_name}.copper_resource.resource_stock": self.copper_stock_df,
            f"{name}.{model_name}.copper_resource.resource_price": self.copper_price_df,
            f"{name}.{model_name}.copper_resource.use_stock": self.copper_use_df,
            f"{name}.{model_name}.copper_resource.recycled_production": self.copper_recycled_production_df,
            f"{name}.{model_name}.platinum_resource.predictable_production": self.platinum_production_df,
            f"{name}.{model_name}.platinum_resource.resource_stock": self.platinum_stock_df,
            f"{name}.{model_name}.platinum_resource.resource_price": self.platinum_price_df,
            f"{name}.{model_name}.platinum_resource.use_stock": self.platinum_use_df,
            f"{name}.{model_name}.platinum_resource.recycled_production": self.platinum_recycled_production_df,
            f"{name}.{model_name}.non_modeled_resource_price": self.non_modeled_resource_df,
        }

        ee.load_study_from_input_dict(inputs_dict)
        ee.execute()
        disc = ee.dm.get_disciplines_with_name(f"{name}.{model_name}")[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        for graph in graph_list:
            graph.to_plotly()  # .show()


if __name__ == "__main__":
    unittest.main()
