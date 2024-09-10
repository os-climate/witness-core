'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/21-2023/11/03 Copyright 2023 Capgemini

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

from os.path import dirname

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore


class GHGEmissionsJacobianDiscTest(AbstractJacobianUnittest):
    # np.set_printoptions(threshold=np.inf)

    def setUp(self):
        self.model_name = 'ghgemission'
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        self.year_start = 2048
        self.year_end = 2053
        years = np.arange(self.year_start, self.year_end + 1)
        self.GHG_total_energy_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                                   GlossaryCore.TotalCO2Emissions: np.linspace(37., 10., len(years)),
                                                   GlossaryCore.TotalN2OEmissions: np.linspace(1.7e-3, 5.e-4,
                                                                                               len(years)),
                                                   GlossaryCore.TotalCH4Emissions: np.linspace(0.17, 0.01, len(years))})
        self.CO2_land_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                           'Crop': np.linspace(0., 0., len(years)),
                                           'Forest': np.linspace(3., 4., len(years))})
        self.CH4_land_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                           'Crop': np.linspace(0., 0., len(years)),
                                           'Forest': np.linspace(3., 4., len(years))})
        self.N2O_land_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                           'Crop': np.linspace(0., 0., len(years)),
                                           'Forest': np.linspace(3., 4., len(years))})

        self.CO2_emissions_ref = 6.49  # Gt

        self.energy_production = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.TotalProductionValue: 100.
        })

        self.residential_energy_consumption = pd.DataFrame({GlossaryCore.Years: years,
                                                               GlossaryCore.TotalProductionValue: 1.2})

        def generate_energy_consumption_df_sector(sector_name):
            out = {GlossaryCore.Years: years}
            out.update({section: 10. for section in GlossaryCore.SectionDictSectors[sector_name]})
            return pd.DataFrame(out)

        self.ghg_eenergy_consumptions_sectors = {f"{self.name}.{sector}.{GlossaryCore.SectionEnergyConsumptionDfValue}": generate_energy_consumption_df_sector(sector) for sector in
                                            GlossaryCore.SectorsPossibleValues}

        self.ghg_non_energy_emissions_sectors = {
            f"{self.name}.{sector}.{GlossaryCore.SectionNonEnergyEmissionGdpDfValue}": generate_energy_consumption_df_sector(
                sector) for sector in
            GlossaryCore.SectorsPossibleValues}

        self.ghg_sections_gdp = {
            f"{self.name}.{sector}.{GlossaryCore.SectionGdpDfValue}": generate_energy_consumption_df_sector(
                sector) for sector in
            GlossaryCore.SectorsPossibleValues}

        self.inputs_cheked = [
            f'{self.name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)}',
            f'{self.name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CH4)}',
            f'{self.name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.N2O)}',
            f'{self.name}.GHG_total_energy_emissions',
            f"{self.name}.{GlossaryCore.EnergyProductionValue}",
        ]

        self.inputs_cheked += [f"{self.name}.{sector}.{GlossaryCore.SectionEnergyConsumptionDfValue}" for sector in GlossaryCore.DefaultSectorListGHGEmissions]
        self.inputs_cheked += [f"{self.name}.{sector}.{GlossaryCore.SectionGdpDfValue}" for sector in GlossaryCore.DefaultSectorListGHGEmissions]

        self.outputs_checked = [
            f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}',
            f'{self.name}.{GlossaryCore.GHGEmissionsDfValue}',
            f"{self.name}.{GlossaryCore.CO2EmissionsObjectiveValue}",
            f"{self.name}.{self.model_name}.{GlossaryCore.EconomicsEmissionDfValue}",
            f"{self.name}.{self.model_name}.{GlossaryCore.EnergyCarbonIntensityDfValue}",
            f"{self.name}.{GlossaryCore.ConstraintCarbonNegative2050}"
        ]

        # self.outputs_checked += [f"{self.name}.{sector}.{GlossaryCore.SectionEnergyEmissionDfValue}" for sector in GlossaryCore.SectorsPossibleValues]
        # self.outputs_checked += [f"{self.name}.{sector}.{GlossaryCore.SectionNonEnergyEmissionDfValue}" for sector in GlossaryCore.SectorsPossibleValues]
        # self.outputs_checked += [f"{self.name}.{sector}.{GlossaryCore.EmissionsDfValue}" for sector in GlossaryCore.SectorsPossibleValues]

    def analytic_grad_entry(self):
        return [
            self.test_carbon_emissions_analytic_grad_affine_co2_objective
        ]

    def test_carbon_emissions_analytic_grad(self):

        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   'ns_agriculture': f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}',
                   GlossaryCore.NS_CCS: f'{self.name}',
                   'ns_energy': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_emissions.ghgemissions.ghgemissions_discipline.GHGemissionsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)}': self.CO2_land_emissions,
                       f'{self.name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CH4)}': self.CH4_land_emissions,
                       f'{self.name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.N2O)}': self.N2O_land_emissions,
                       f'{self.name}.GHG_total_energy_emissions': self.GHG_total_energy_emissions,
                       f"{self.name}.{GlossaryCore.CO2EmissionsRef['var_name']}": self.CO2_emissions_ref,
                       f"{self.name}.{GlossaryCore.EnergyProductionValue}": self.energy_production,
                       f"{self.name}.{GlossaryCore.ResidentialEnergyConsumptionDfValue}": self.residential_energy_consumption,
                       **self.ghg_eenergy_consumptions_sectors,
                       **self.ghg_sections_gdp,
                       **self.ghg_non_energy_emissions_sectors,
                       }

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_ghg_emission_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data=disc_techno.local_data,
                            inputs=self.inputs_cheked,
                            outputs=self.outputs_checked)

    def test_carbon_emissions_analytic_grad_affine_co2_objective(self):

        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   'ns_agriculture': f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}',
                   GlossaryCore.NS_CCS: f'{self.name}',
                   'ns_energy': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_emissions.ghgemissions.ghgemissions_discipline.GHGemissionsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)}': self.CO2_land_emissions,
                       f'{self.name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CH4)}': self.CH4_land_emissions,
                       f'{self.name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.N2O)}': self.N2O_land_emissions,
                       f'{self.name}.GHG_total_energy_emissions': self.GHG_total_energy_emissions,
                       f"{self.name}.{GlossaryCore.CO2EmissionsRef['var_name']}": self.CO2_emissions_ref,
                       f"{self.name}.affine_co2_objective": False,
                       f"{self.name}.{GlossaryCore.EnergyProductionValue}": self.energy_production,
                       f"{self.name}.{GlossaryCore.ResidentialEnergyConsumptionDfValue}": self.residential_energy_consumption,
                       **self.ghg_eenergy_consumptions_sectors,
                       **self.ghg_sections_gdp,
                       **self.ghg_non_energy_emissions_sectors,
                       }

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_ghg_emission_discipline_affine_co2_objective.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data=disc_techno.local_data,
                            inputs=self.inputs_cheked,
                            outputs=self.outputs_checked)
