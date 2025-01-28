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

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.glossarycore import GlossaryCore


class GHGEmissionDiscTest(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_execute(self):

        self.model_name = 'ghgemission'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_CCS: f'{self.name}',
                   'ns_energy': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}',
                   }

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_emissions.ghgemissions.ghgemissions_discipline.GHGemissionsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        year_start = GlossaryCore.YearStartDefault
        year_end = GlossaryCore.YearEndDefaultTest
        years = np.arange(year_start, year_end + 1)
        GHG_total_energy_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                                   GlossaryCore.TotalCO2Emissions: np.linspace(37., 10., len(years)),
                                                   GlossaryCore.TotalN2OEmissions: np.linspace(1.7e-3, 5.e-4, len(years)),
                                                   GlossaryCore.TotalCH4Emissions: np.linspace(0.17, 0.01, len(years))})
        CO2_land_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                           'Crop': np.linspace(0., 0., len(years)),
                                           'Forest': np.linspace(3., 4., len(years))})
        N2O_land_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                           'Crop': np.linspace(0., 0., len(years)),
                                           'Forest': np.linspace(3., 4., len(years))})
        CH4_land_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                           'Crop': np.linspace(0., 0., len(years)),
                                           'Forest': np.linspace(3., 4., len(years))})

        CO2_emissions_ref = 6.49 # Gt

        energy_production = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.TotalProductionValue: 84.
        })

        residential_energy_consumption = pd.DataFrame({GlossaryCore.Years: years,
                                                       GlossaryCore.TotalProductionValue: 1.2})

        all_sections = []
        for sector in GlossaryCore.SectorsPossibleValues:
            all_sections.extend(GlossaryCore.SectionDictSectors[sector])

        def generate_energy_consumption_df_sector(sector_name):
            out = {GlossaryCore.Years: years}
            out.update({section: 10. for section in all_sections})
            return pd.DataFrame(out)

        ghg_eenergy_consumptions_sectors = {
            f"{self.name}.{sector}.{GlossaryCore.SectionEnergyConsumptionDfValue}": generate_energy_consumption_df_sector(sector) for sector in
                                            GlossaryCore.SectorsPossibleValues}

        ghg_non_energy_emissions_sectors = {
            f"{self.name}.{sector}.{GlossaryCore.SectionNonEnergyEmissionGdpDfValue}": generate_energy_consumption_df_sector(
                sector) for sector in
            GlossaryCore.SectorsPossibleValues}

        ghg_sections_gdp = {
            f"{self.name}.{sector}.{GlossaryCore.SectionGdpDfValue}": generate_energy_consumption_df_sector(
                sector) for sector in
            GlossaryCore.SectorsPossibleValues}

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': year_end,
                       f'{self.name}.{GlossaryCore.SectorListValue}': GlossaryCore.SectorsPossibleValues,
                       f'{self.name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)}': CO2_land_emissions,
                       f'{self.name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CH4)}': CH4_land_emissions,
                       f'{self.name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.N2O)}': N2O_land_emissions,
                       f'{self.name}.GHG_total_energy_emissions': GHG_total_energy_emissions,
                       f"{self.name}.{GlossaryCore.CO2EmissionsRef['var_name']}": CO2_emissions_ref,
                       f"{self.name}.{GlossaryCore.StreamProductionValue}": energy_production,
                       f"{self.name}.{GlossaryCore.ResidentialEnergyConsumptionDfValue}": residential_energy_consumption,

                       **ghg_eenergy_consumptions_sectors,
                       **ghg_non_energy_emissions_sectors,
                       **ghg_sections_gdp
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