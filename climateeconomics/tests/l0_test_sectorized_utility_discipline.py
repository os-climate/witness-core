'''
Copyright 2022 Airbus SAS
Modifications on 2023/08/17-2023/11/03 Copyright 2023 Capgemini

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
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.glossarycore import GlossaryCore


class SectorizedUtilityDiscTest(unittest.TestCase):
    np.set_printoptions(threshold=np.inf)

    def setUp(self):
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        self.years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1)
        self.economics_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.OutputNetOfDamage: np.linspace(121, 91, len(self.years)),
            GlossaryCore.GrossOutput: 0.,
            GlossaryCore.PerCapitaConsumption: 0.,
        })

        self.sector_list = GlossaryCore.SectorsPossibleValues

    def test_execute(self):
        self.model_name = GlossaryCore.Consumption
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   GlossaryCore.NS_SECTORS: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.sectorized_utility.sectorized_utility_discipline.SectorizedUtilityDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')

        population_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.PopulationValue: np.linspace(7886, 9550, len(self.years))
        })

        # put manually the index
        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        energy_price = np.arange(200, 200 + len(years))
        energy_mean_price = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.EnergyPriceValue: energy_price})
        residential_energy_conso_ref = 21
        residential_energy = np.linspace(21, 15, len(years))
        residential_energy_df = pd.DataFrame(
            {GlossaryCore.Years: years, GlossaryCore.TotalProductionValue: residential_energy})
        # Share invest
        invest = np.asarray([10] * len(years))
        investment_df = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.InvestmentsValue: invest})
        np.set_printoptions(threshold=np.inf)

        # Sectorized Consumption
        sectorized_consumption_df = pd.DataFrame({GlossaryCore.Years: years})
        for sector in self.sector_list:
            sectorized_consumption_df[sector] = 1.0

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': GlossaryCore.YearStartDefault,
                       f'{self.name}.{GlossaryCore.YearEnd}': GlossaryCore.YearEndDefault,
                       f'{self.name}.{GlossaryCore.TimeStep}': 1,
                       f'{self.name}.conso_elasticity': 1.45,
                       f'{self.name}.init_rate_time_pref': 0.015,
                       f'{self.name}.{GlossaryCore.EconomicsDfValue}': self.economics_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': population_df,
                       f'{self.name}.{GlossaryCore.EnergyMeanPriceValue}': energy_mean_price,
                       f'{self.name}.residential_energy_conso_ref': residential_energy_conso_ref,
                       f'{self.name}.{GlossaryCore.ResidentialEnergyConsumptionDfValue}': residential_energy_df,
                       f'{self.name}.{GlossaryCore.InvestmentDfValue}': investment_df,
                       f'{self.name}.{GlossaryCore.AllSectorsDemandDfValue}': sectorized_consumption_df,
                       }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        for graph in graph_list:
            graph.to_plotly().show()
