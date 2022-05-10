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
from sos_trades_core.tools.post_processing.post_processing_factory import PostProcessingFactory


class PolicyDiscTest(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_execute(self):

        self.model_name = 'policy'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.policymodel.policy_discipline.PolicyDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        years = np.arange(2020, 2101)
        CCS_price = pd.DataFrame(
            {'years': years, 'ccs_price_per_tCO2': np.linspace(311, 515, len(years))})
        CO2_damage = pd.DataFrame(
            {'years': years, 'CO2_damage_price': np.linspace(345.5, 433.2,  len(years))})

        values_dict = {f'{self.name}.CCS_price': CCS_price,
                       f'{self.name}.CO2_damage_price': CO2_damage,
                       f'{self.name}.ccs_price_percentage': 50.,
                       f'{self.name}.co2_damage_price_percentage': 50.}

        self.ee.dm.set_values_from_dict(values_dict)

        self.ee.execute()

        max_CCS_CO2 = np.maximum(
            CCS_price['ccs_price_per_tCO2'], CO2_damage['CO2_damage_price'])
        CO2_tax = self.ee.dm.get_value(f'{self.name}.CO2_taxes')

#         np.testing.assert_almost_equal(
# max_CCS_CO2, CO2_tax['CO2_tax'].values, err_msg='arrays are not equal')

        ppf = PostProcessingFactory()
        disc = self.ee.dm.get_disciplines_with_name(f'Test.policy')
        filters = ppf.get_post_processing_filters_by_discipline(
            disc[0])
        graph_list = ppf.get_post_processing_by_discipline(
            disc[0], filters, as_json=False)

#         for graph in graph_list:
#             graph.to_plotly().show()
