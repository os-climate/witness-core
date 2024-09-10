'''
Copyright 2023 Capgemini

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


class SectorsRedistributionEnergyDisciplineTest(unittest.TestCase):

    def setUp(self):
        """Initialize third data needed for testing"""
        self.year_start =GlossaryCore.YearStartDefault
        self.year_end = 2050
        self.years = np.arange(self.year_start, self.year_end + 1)
        n_years = len(self.years)
        time = self.years - self.year_start
        self.sector_list = GlossaryCore.SectorsPossibleValues

        self.energy_production_df = pd.DataFrame({GlossaryCore.Years: self.years,
                                                  GlossaryCore.TotalProductionValue: 10000 * 1.02 ** time})
        self.share_energy_agriculture = pd.DataFrame({GlossaryCore.Years: self.years,
                                                      GlossaryCore.ShareSectorEnergy: np.linspace(5, 2, n_years)})
        self.share_energy_services = pd.DataFrame({GlossaryCore.Years: self.years,
                                                   GlossaryCore.ShareSectorEnergy: np.linspace(30, 35, n_years)})
        self.share_energy_residential = pd.DataFrame({GlossaryCore.Years: self.years,
                                                   GlossaryCore.ShareSectorEnergy: np.linspace(20, 25, n_years)})
        self.share_energy_other = pd.DataFrame({GlossaryCore.Years: self.years,
                                                   GlossaryCore.ShareSectorEnergy: np.linspace(10, 10, n_years)})

    def test(self):
        """Check discipline setup and run"""
        name = 'Test'
        model_name = 'sectors_redistribution_energy.SectorsRedistributionEnergyDiscipline'
        ee = ExecutionEngine(name)
        ns_dict = {'ns_public': f'{name}',
                   GlossaryCore.NS_WITNESS: f'{name}',
                   GlossaryCore.NS_FUNCTIONS: f'{name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{name}',
                   'ns_coal_resource': f'{name}',
                   'ns_resource': f'{name}',
                   GlossaryCore.NS_SECTORS: f'{name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{name}',
                   }
        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.sectors_redistribution_energy.sectors_redistribution_energy_discipline.SectorsRedistributionEnergyDiscipline'
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()

        inputs_dict = {f'{name}.{GlossaryCore.EnergyProductionValue}': self.energy_production_df,
                       f'{name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.ShareSectorEnergyDfValue}': self.share_energy_agriculture,
                       f'{name}.{GlossaryCore.SectorServices}.{GlossaryCore.ShareSectorEnergyDfValue}': self.share_energy_services,
                       f'{name}.{GlossaryCore.ShareResidentialEnergyDfValue}': self.share_energy_residential,
                       f'{name}.{model_name}.{GlossaryCore.ShareOtherEnergyDfValue}': self.share_energy_other,
        }
        ee.load_study_from_input_dict(inputs_dict)

        ee.execute()

        disc = ee.root_process.proxy_disciplines[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass

