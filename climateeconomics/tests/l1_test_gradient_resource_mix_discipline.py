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
from os.path import dirname, join

import numpy as np
import pandas as pd
from pandas import read_csv
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.core.core_resources.resource_mix.resource_mix import (
    ResourceMixModel,
)
from climateeconomics.glossarycore import GlossaryCore


class ResourceJacobianDiscTest(AbstractJacobianUnittest):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        self.model_name = 'all_resource'

        data_dir = join(dirname(__file__), 'data')
        self.oil_production_df = read_csv(
            join(data_dir, 'oil.predictible_production.csv'))
        self.oil_production_df.set_index(GlossaryCore.Years, inplace=True)
        self.gas_production_df = read_csv(
            join(data_dir, 'gas.predictible_production.csv'))
        self.coal_production_df = read_csv(
            join(data_dir, 'coal.predictible_production.csv'))
        self.uranium_production_df = read_csv(
            join(data_dir, 'uranium.predictible_production.csv'))
        self.copper_production_df = read_csv(
            join(data_dir, 'copper.predictible_production.csv'))
        self.platinum_production_df = read_csv(
            join(data_dir, 'platinum.predictible_production.csv'))
        self.oil_stock_df = read_csv(
            join(data_dir, 'oil.stock.csv'))
        self.gas_stock_df = read_csv(
            join(data_dir, 'gas.stock.csv'))
        self.uranium_stock_df = read_csv(
            join(data_dir, 'uranium.stock.csv'))
        self.coal_stock_df = read_csv(
            join(data_dir, 'coal.stock.csv'))
        self.copper_stock_df = read_csv(
            join(data_dir, 'copper.stock.csv'))
        self.platinum_stock_df = read_csv(
            join(data_dir, 'platinum.stock.csv'))
        self.oil_price_df = read_csv(
            join(data_dir, 'oil.price.csv'))
        self.gas_price_df = read_csv(
            join(data_dir, 'gas.price.csv'))
        self.coal_price_df = read_csv(
            join(data_dir, 'coal.price.csv'))
        self.uranium_price_df = read_csv(
            join(data_dir, 'uranium.price.csv'))
        self.copper_price_df = read_csv(
            join(data_dir, 'copper.price.csv'))
        self.platinum_price_df = read_csv(
            join(data_dir, 'platinum.price.csv'))
        self.oil_use_df = read_csv(
            join(data_dir, 'oil.use.csv'))
        self.gas_use_df = read_csv(
            join(data_dir, 'gas.use.csv'))
        self.coal_use_df = read_csv(
            join(data_dir, 'coal.use.csv'))
        self.uranium_use_df = read_csv(
            join(data_dir, 'uranium.use.csv'))
        self.copper_use_df = read_csv(
            join(data_dir, 'copper.use.csv'))
        self.platinum_use_df = read_csv(
            join(data_dir, 'platinum.use.csv'))
        self.oil_recycled_production_df = read_csv(
            join(data_dir, 'oil.recycled_production.csv'))
        self.gas_recycled_production_df = read_csv(
            join(data_dir, 'gas.recycled_production.csv'))
        self.coal_recycled_production_df = read_csv(
            join(data_dir, 'coal.recycled_production.csv'))
        self.uranium_recycled_production_df = read_csv(
            join(data_dir, 'uranium.recycled_production.csv'))
        self.copper_recycled_production_df = read_csv(
            join(data_dir, 'copper.recycled_production.csv'))
        self.platinum_recycled_production_df = read_csv(
            join(data_dir, 'platinum.recycled_production.csv'))
        self.non_modeled_resource_df = read_csv(
            join(data_dir, 'resource_data_price.csv'))
        self.all_demand = read_csv(
            join(data_dir, 'all_demand_with_high_demand.csv'))

        self.year_start = 2020
        self.year_end = GlossaryCore.YearEndDefaultTest
        self.years = np.arange(self.year_start, self.year_end + 1, 1)
        self.year_range = self.year_end - self.year_start + 1

        self.values_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                            f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                            f'{self.name}.{self.model_name}.resources_demand': self.all_demand,
                            f'{self.name}.{self.model_name}.resources_demand_woratio': self.all_demand,
                            f'{self.name}.{self.model_name}.oil_resource.predictable_production': self.oil_production_df,
                            f'{self.name}.{self.model_name}.oil_resource.resource_stock': self.oil_stock_df,
                            f'{self.name}.{self.model_name}.oil_resource.resource_price': self.oil_price_df,
                            f'{self.name}.{self.model_name}.oil_resource.use_stock': self.oil_use_df,
                            f'{self.name}.{self.model_name}.oil_resource.recycled_production': self.oil_recycled_production_df,
                            f'{self.name}.{self.model_name}.natural_gas_resource.predictable_production': self.gas_production_df,
                            f'{self.name}.{self.model_name}.natural_gas_resource.resource_stock': self.gas_stock_df,
                            f'{self.name}.{self.model_name}.natural_gas_resource.resource_price': self.gas_price_df,
                            f'{self.name}.{self.model_name}.natural_gas_resource.use_stock': self.gas_use_df,
                            f'{self.name}.{self.model_name}.natural_gas_resource.recycled_production': self.gas_recycled_production_df,
                            f'{self.name}.{self.model_name}.coal_resource.predictable_production': self.coal_production_df,
                            f'{self.name}.{self.model_name}.coal_resource.resource_stock': self.coal_stock_df,
                            f'{self.name}.{self.model_name}.coal_resource.resource_price': self.coal_price_df,
                            f'{self.name}.{self.model_name}.coal_resource.use_stock': self.coal_use_df,
                            f'{self.name}.{self.model_name}.coal_resource.recycled_production': self.coal_recycled_production_df,
                            f'{self.name}.{self.model_name}.uranium_resource.predictable_production': self.uranium_production_df,
                            f'{self.name}.{self.model_name}.uranium_resource.resource_stock': self.uranium_stock_df,
                            f'{self.name}.{self.model_name}.uranium_resource.resource_price': self.uranium_price_df,
                            f'{self.name}.{self.model_name}.uranium_resource.use_stock': self.uranium_use_df,
                            f'{self.name}.{self.model_name}.uranium_resource.recycled_production': self.uranium_recycled_production_df,
                            f'{self.name}.{self.model_name}.platinum_resource.predictable_production': self.platinum_production_df,
                            f'{self.name}.{self.model_name}.platinum_resource.resource_stock': self.platinum_stock_df,
                            f'{self.name}.{self.model_name}.platinum_resource.resource_price': self.platinum_price_df,
                            f'{self.name}.{self.model_name}.platinum_resource.use_stock': self.platinum_use_df,
                            f'{self.name}.{self.model_name}.platinum_resource.recycled_production': self.platinum_recycled_production_df,
                            f'{self.name}.{self.model_name}.{ResourceMixModel.NON_MODELED_RESOURCE_PRICE}': self.non_modeled_resource_df
                            }

        self.ns_dict = {'ns_public': f'{self.name}',
                        'ns_coal_resource': f'{self.name}.{self.model_name}',
                        'ns_oil_resource': f'{self.name}.{self.model_name}',
                        'ns_natural_gas_resource': f'{self.name}.{self.model_name}',
                        'ns_uranium_resource': f'{self.name}.{self.model_name}',
                        'ns_copper_resource': f'{self.name}.{self.model_name}',
                        'ns_platinum_resource': f'{self.name}.{self.model_name}',
                        'ns_resource': f'{self.name}.{self.model_name}',
                        }

    def analytic_grad_entry(self):
        return [
            self._test_All_resource_discipline_analytic_grad
        ]

    def test_All_resource_discipline_analytic_grad(self):

        self.ee.ns_manager.add_ns_def(self.ns_dict)

        mod_path = 'climateeconomics.core.core_resources.resource_mix.resource_mix_disc.ResourceMixDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes(True)

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{self.model_name}.resources_demand': self.all_demand,
                       f'{self.name}.{self.model_name}.resources_demand_woratio': self.all_demand,
                       f'{self.name}.{self.model_name}.oil_resource.predictable_production': self.oil_production_df,
                       f'{self.name}.{self.model_name}.oil_resource.resource_stock': self.oil_stock_df,
                       f'{self.name}.{self.model_name}.oil_resource.resource_price': self.oil_price_df,
                       f'{self.name}.{self.model_name}.oil_resource.use_stock': self.oil_use_df,
                       f'{self.name}.{self.model_name}.oil_resource.recycled_production': self.oil_recycled_production_df,
                       f'{self.name}.{self.model_name}.natural_gas_resource.predictable_production': self.gas_production_df,
                       f'{self.name}.{self.model_name}.natural_gas_resource.resource_stock': self.gas_stock_df,
                       f'{self.name}.{self.model_name}.natural_gas_resource.resource_price': self.gas_price_df,
                       f'{self.name}.{self.model_name}.natural_gas_resource.use_stock': self.gas_use_df,
                       f'{self.name}.{self.model_name}.natural_gas_resource.recycled_production': self.gas_recycled_production_df,
                       f'{self.name}.{self.model_name}.coal_resource.predictable_production': self.coal_production_df,
                       f'{self.name}.{self.model_name}.coal_resource.resource_stock': self.coal_stock_df,
                       f'{self.name}.{self.model_name}.coal_resource.resource_price': self.coal_price_df,
                       f'{self.name}.{self.model_name}.coal_resource.use_stock': self.coal_use_df,
                       f'{self.name}.{self.model_name}.coal_resource.recycled_production': self.coal_recycled_production_df,
                       f'{self.name}.{self.model_name}.uranium_resource.predictable_production': self.uranium_production_df,
                       f'{self.name}.{self.model_name}.uranium_resource.resource_stock': self.uranium_stock_df,
                       f'{self.name}.{self.model_name}.uranium_resource.resource_price': self.uranium_price_df,
                       f'{self.name}.{self.model_name}.uranium_resource.use_stock': self.uranium_use_df,
                       f'{self.name}.{self.model_name}.uranium_resource.recycled_production': self.uranium_recycled_production_df,
                       f'{self.name}.{self.model_name}.copper_resource.predictable_production': self.copper_production_df,
                       f'{self.name}.{self.model_name}.copper_resource.resource_stock': self.copper_stock_df,
                       f'{self.name}.{self.model_name}.copper_resource.resource_price': self.copper_price_df,
                       f'{self.name}.{self.model_name}.copper_resource.use_stock': self.copper_use_df,
                       f'{self.name}.{self.model_name}.copper_resource.recycled_production': self.copper_recycled_production_df,
                       f'{self.name}.{self.model_name}.platinum_resource.predictable_production': self.platinum_production_df,
                       f'{self.name}.{self.model_name}.platinum_resource.resource_stock': self.platinum_stock_df,
                       f'{self.name}.{self.model_name}.platinum_resource.resource_price': self.platinum_price_df,
                       f'{self.name}.{self.model_name}.platinum_resource.use_stock': self.platinum_use_df,
                       f'{self.name}.{self.model_name}.platinum_resource.recycled_production': self.platinum_recycled_production_df,
                       f'{self.name}.{self.model_name}.{ResourceMixModel.NON_MODELED_RESOURCE_PRICE}': self.non_modeled_resource_df
                       }
        for name, value in values_dict.items():
            if isinstance(value, pd.DataFrame) and GlossaryCore.Years in value.columns:
                values_dict[name] = value.loc[value[GlossaryCore.Years] <= self.year_end]
        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline
        input_names = []
        input_stock = [
            f'{self.name}.{self.model_name}.{resource}.resource_stock' for resource in ResourceMixModel.RESOURCE_LIST]
        input_names.extend(input_stock)
        # input_use_stock = [
        #     f'{self.name}.{self.model_name}.{resource}.use_stock' for resource in ResourceMixModel.RESOURCE_LIST]
        # input_names.extend(input_use_stock)
        input_recycled_production = [
            f'{self.name}.{self.model_name}.{resource}.recycled_production' for resource in ResourceMixModel.RESOURCE_LIST]
        input_names.extend(input_recycled_production)
        input_price = [
            f'{self.name}.{self.model_name}.{resource}.resource_price' for resource in ResourceMixModel.RESOURCE_LIST]
        input_names.extend(input_price)
        input_other_price = [
            f'{self.name}.{self.model_name}.{ResourceMixModel.NON_MODELED_RESOURCE_PRICE}']
        input_names.extend(input_other_price)
        input_demand = [
            f'{self.name}.{self.model_name}.resources_demand']
        input_demand.extend(
            [f'{self.name}.{self.model_name}.resources_demand_woratio'])
        input_names.extend(input_demand)
        resource_output = [
                            f'{self.name}.{self.model_name}.{ResourceMixModel.ALL_RESOURCE_STOCK}',
                            f'{self.name}.{self.model_name}.{ResourceMixModel.All_RESOURCE_USE}',
                            f'{self.name}.{self.model_name}.{GlossaryCore.ResourcesPriceValue}',
                            f'{self.name}.{self.model_name}.{ResourceMixModel.ALL_RESOURCE_RECYCLED_PRODUCTION}',
                            f'{self.name}.{self.model_name}.{ResourceMixModel.RATIO_USABLE_DEMAND}',
                            f'{self.name}.{self.model_name}.{ResourceMixModel.ALL_RESOURCE_DEMAND}',
                           ]

        self.override_dump_jacobian = 1
        self.check_jacobian(location=dirname(__file__), filename='jacobian_all_resource_discipline.pkl',
                            discipline=disc_techno, local_data = disc_techno.local_data, inputs=input_names,
                            outputs=resource_output, step=1e-15,
                            derr_approx='complex_step')
if __name__ == "__main__":
    unittest.main()