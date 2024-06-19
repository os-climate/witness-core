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

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class IEADataPreparationTest(unittest.TestCase):

    def setUp(self):
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_execute(self):
        self.model_name = 'IEADataPreparation'
        year_start = 2020
        year_end = 2055
        years = [2023, 2030, 2040, 2050]

        ns_dict = {'ns_public': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.post_procs.iea_data_preparation.iea_data_preparation_discipline.IEADataPreparationDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()
        CO2_emissions_df = pd.DataFrame({GlossaryCore.Years: years,
                                         GlossaryCore.TotalCO2Emissions: [60, 40, 20, 30]})
        GDP_df = pd.DataFrame({GlossaryCore.Years: years,
                               GlossaryCore.OutputNetOfDamage: [120, 140, 145, 160]
                               })
        CO2_tax_df = pd.DataFrame({GlossaryCore.Years: years,
                                   GlossaryCore.CO2Tax: [100, 500, 700, 800]})

        energy_production_df = pd.DataFrame({GlossaryCore.Years: years,
                                             GlossaryCore.TotalProductionValue: [40, 70, 80, 10]})

        population_df = pd.DataFrame({GlossaryCore.Years: years,
                                      GlossaryCore.PopulationValue: [8, 8.2, 8.3, 8]})

        temperature_df = pd.DataFrame({GlossaryCore.Years: years,
                                      GlossaryCore.TempAtmo: [2.2, 2.7, 2.75, 2.78]})

        values_dict = {
            f'{self.name}.{GlossaryCore.YearStart}': year_start,
            f'{self.name}.{GlossaryCore.YearEnd}': year_end,
            f'{self.name}.{self.model_name}.{GlossaryCore.CO2EmissionsGtValue}': CO2_emissions_df,
            f'{self.name}.{self.model_name}.{GlossaryCore.EconomicsDfValue}': GDP_df,
            f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxesValue}': CO2_tax_df,
            f'{self.name}.{self.model_name}.{GlossaryCore.EnergyProductionValue}': energy_production_df,
            f'{self.name}.{self.model_name}.{GlossaryCore.TemperatureDfValue}': temperature_df,
            f'{self.name}.{self.model_name}.{GlossaryCore.PopulationDfValue}': population_df,
        }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        for graph in graph_list:
            # graph.to_plotly().show()
            pass
