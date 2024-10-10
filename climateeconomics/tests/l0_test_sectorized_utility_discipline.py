'''
Copyright 2024 Capgemini

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

from climateeconomics.glossarycore import GlossaryCore


class SectorizedUtilityDiscTest(unittest.TestCase):
    np.set_printoptions(threshold=np.inf)

    def setUp(self):
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = GlossaryCore.YearEndDefaultTest
        self.years = np.arange(self.year_start, self.year_end + 1)
        self.sector_list = GlossaryCore.SectorsPossibleValues

    def test_execute(self):
        self.model_name = GlossaryCore.Consumption
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_SECTORS: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.utility.sectorized_utility_discipline.SectorizedUtilityDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        population_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.PopulationValue: np.linspace(7886, 9550, len(self.years))
        })

        energy_mean_price = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.EnergyPriceValue: np.arange(200, 200 + len(self.years))
        })
        economics_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.GrossOutput: 10.,
            GlossaryCore.OutputNetOfDamage: 0.,
            GlossaryCore.PerCapitaConsumption: 0.,
            GlossaryCore.Capital: 0.,
        })

        sectorized_consumption_df = pd.DataFrame({GlossaryCore.Years: self.years})
        for sector in self.sector_list:
            sectorized_consumption_df[sector] = np.linspace(np.random.uniform(.7, 1.3), np.random.uniform(.7, 1.3), len(self.years))

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': GlossaryCore.YearStartDefault,
                       f'{self.name}.{GlossaryCore.YearEnd}': GlossaryCore.YearEndDefault,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': population_df,
                       f'{self.name}.{GlossaryCore.EconomicsDfValue}': economics_df,
                       f'{self.name}.{GlossaryCore.EnergyMeanPriceValue}': energy_mean_price,
                       f'{self.name}.{GlossaryCore.SectorizedConsumptionDfValue}': sectorized_consumption_df,
                       }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        for graph in graph_list:
           #graph.to_plotly().show()
           pass
