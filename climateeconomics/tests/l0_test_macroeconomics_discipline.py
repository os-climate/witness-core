'''
Copyright 2022 Airbus SAS
Modifications on 2023/08/17-2023/11/09 Copyright 2023 Capgemini

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
from os.path import dirname, join

import numpy as np
import pandas as pd
from pandas import DataFrame
from sostrades_optimization_plugins.models.test_class import GenericDisciplinesTestClass

from climateeconomics.glossarycore import GlossaryCore


class MacroDiscTest(GenericDisciplinesTestClass):
    """Energy Market discipline test class"""

    def setUp(self):
        self.name = 'Test'
        self.model_name = 'Energy market'
        self.mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.macroeconomics.macroeconomics_discipline.MacroeconomicsDiscipline'
        self.ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                        GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                        'ns_public': f'{self.name}',
                        'ns_energy_market': f'{self.name}',
                        GlossaryCore.NS_MACRO: f'{self.name}',
                        GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                        GlossaryCore.NS_GHGEMISSIONS: f'{self.name}',
                        'ns_energy_study': f'{self.name}', }
        self.pickle_prefix = self.model_name
        self.jacobian_test = False
        self.show_graphs = False
        self.override_dump_jacobian = False
        self.pickle_directory = dirname(__file__)


        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = GlossaryCore.YearEndDefault
        self.years = np.arange(self.year_start, self.year_end + 1)
        self.nb_years = len(self.years)
        self.energy_supply_df_all = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalCO2Emissions: np.linspace(35, 0, len(self.years))
        })

        self.co2_emissions_gt = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalCO2Emissions: np.linspace(1035, 0, len(self.years)),
        })

        self.energy_investment_wo_tax = DataFrame(
            {GlossaryCore.Years: self.years,
             GlossaryCore.EnergyInvestmentsWoTaxValue: 3.5})  # in G$

        self.share_non_energy_investment = DataFrame(
            {GlossaryCore.Years: self.years,
             GlossaryCore.ShareNonEnergyInvestmentsValue: 27. - 2.6})

        self.energy_supply_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            "Total": np.linspace(115, 145, len(self.years))
        })

        self.damage_fraction_df = pd.DataFrame({GlossaryCore.Years: self.years,
                                                GlossaryCore.DamageFractionOutput: np.linspace(0.01, 0.05,
                                                                                               self.nb_years),
                                                })

        self.energy_market_ratios = pd.DataFrame({
            GlossaryCore.Years: self.years,
            "Total": 85.,
        })

        self.default_CO2_tax = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.CO2Tax: 50.0})

        # energy_capital
        nb_years = len(self.years)
        energy_capital_year_start = 16.09
        energy_capital = []
        energy_capital.append(energy_capital_year_start)
        for year in np.arange(1, nb_years):
            energy_capital.append(energy_capital[year - 1] * 1.02)
        self.energy_capital_df = pd.DataFrame({GlossaryCore.Years: self.years, GlossaryCore.Capital: energy_capital})

        # retrieve co2_emissions_gt input
        global_data_dir = join(dirname(dirname(__file__)), 'data')

        self.population_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.PopulationValue: np.linspace(7886, 9550, len(self.years))
        })

        self.working_age_pop_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.Population1570: np.linspace(5490, 6061, len(self.years))
        })

        self.share_energy_per_sector_percentage = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.SectorServices: 14.56,
            GlossaryCore.SectorAgriculture: 2.13,
            GlossaryCore.SectorIndustry: 63.30,
            # GlossaryCore.SectorNonEco: 20.0, # not an economic sector
        })

        self.carbon_intensity_energy = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.EnergyCarbonIntensityDfValue: 10.
        })

        self.default_co2_efficiency = pd.DataFrame(
            {GlossaryCore.Years: self.years, GlossaryCore.CO2TaxEfficiencyValue: 40.0})
        self.sectors_list = [GlossaryCore.SectorServices, GlossaryCore.SectorAgriculture, GlossaryCore.SectorIndustry]
        self.section_list = GlossaryCore.SectionsPossibleValues
        global_data_dir = join(dirname(dirname(__file__)), 'data')
        weighted_average_percentage_per_sector_df = pd.read_csv(
            join(global_data_dir, 'weighted_average_percentage_per_sector.csv'))
        subsector_share_dict = {
            **{GlossaryCore.Years: self.years, },
            **dict(zip(weighted_average_percentage_per_sector_df.columns[1:],
                       weighted_average_percentage_per_sector_df.values[0, 1:]))
        }
        self.section_gdp_df = pd.DataFrame(subsector_share_dict)
        self.share_residential_energy_consumption = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.ShareSectorEnergy: 10.
        })

    def get_inputs_dict(self) -> dict:
        return {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
         f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
         f'{self.name}.init_rate_time_pref': 0.015,
         f'{self.name}.conso_elasticity': 1.45,
         f'{self.name}.{self.model_name}.{GlossaryCore.DamageToProductivity}': True,
         f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': self.energy_investment_wo_tax,
         f'{self.name}.{GlossaryCore.ShareNonEnergyInvestmentsValue}': self.share_non_energy_investment,
         f'{self.name}.{GlossaryCore.EnergyMixNetProductionsDfValue}': self.energy_supply_df,
         f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
         f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
         f'{self.name}.{GlossaryCore.CO2TaxesValue}': self.default_CO2_tax,
         f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': self.default_co2_efficiency,
         f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': self.co2_emissions_gt,
         f'{self.name}.{GlossaryCore.WorkingAgePopulationDfValue}': self.working_age_pop_df,
         f'{self.name}.{GlossaryCore.EnergyCapitalDfValue}': self.energy_capital_df,
         f'{self.name}.{GlossaryCore.SectorListValue}': self.sectors_list,
         f'{self.name}.{GlossaryCore.SectionList}': self.section_list,
         f'{self.name}.{GlossaryCore.SectionGdpPercentageDfValue}': self.section_gdp_df,
         f'{self.name}.{GlossaryCore.ShareResidentialEnergyDfValue}': self.share_residential_energy_consumption,
         f'{self.name}.assumptions_dict': {
             'compute_gdp': True,
             'compute_climate_impact_on_gdp': True,
             'activate_climate_effect_population': True,
             'activate_pandemic_effects': True,
         },
         f'{self.name}.{GlossaryCore.EnergyCarbonIntensityDfValue}': self.carbon_intensity_energy,
         f'{self.name}.{GlossaryCore.EnergyMarketRatioAvailabilitiesValue}': self.energy_market_ratios,
         }

    def test_execute(self):
        self.model_name = 'Macroeconomics'
