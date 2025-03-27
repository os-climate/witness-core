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
from os.path import dirname

import numpy as np
import pandas as pd
from sostrades_optimization_plugins.models.test_class import GenericDisciplinesTestClass

from climateeconomics.glossarycore import GlossaryCore


class ServicesDiscTest(GenericDisciplinesTestClass):
    """Energy Market discipline test class"""

    def setUp(self):
        self.name = 'Test'
        self.sector_name = GlossaryCore.SectorIndustry
        self.model_name = self.sector_name
        self.pickle_prefix = self.model_name
        self.show_graphs = False
        self.override_dump_jacobian = False
        self.jacobian_test = False
        self.pickle_directory = dirname(__file__)

        self.ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_MACRO: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   "ns_energy_market": f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_SECTORS: f'{self.name}'}


        self.mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.sector_discipline.SectorDiscipline'

        # put manually the index
        self.years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = GlossaryCore.YearEndDefault

        # input
        self.workforce_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            self.sector_name: np.linspace(5490, 6061, len(self.years)) * 0.659 * 0.509
        })

        self.energy_supply_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalProductionValue: np.linspace(43, 76, len(self.years))
        })

        self.total_invest = pd.DataFrame({GlossaryCore.Years: self.years,
                                          GlossaryCore.InvestmentsValue: 5 * 1.02 ** np.arange(len(self.years))})
        
        #damage
        self.damage_fraction_df = pd.DataFrame({GlossaryCore.Years: self.years,
                                                GlossaryCore.DamageFractionOutput: np.linspace(0.02, 0.05, len(self.years)),})

        self.section_list = GlossaryCore.SectionsIndustry
        self.energy_market_ratios = pd.DataFrame({
            GlossaryCore.Years: self.years,
            "Total": 85.,
        })

    def get_inputs_dict(self) -> dict:
        return {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                f'{self.name}.{GlossaryCore.DamageToProductivity}': True,
                f'{self.name}.{self.sector_name}.{GlossaryCore.InvestmentDfValue}': self.total_invest,
                f'{self.name}.{self.sector_name}.{GlossaryCore.StreamProductionValue}': self.energy_supply_df,
                f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction_df,
                f'{self.name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df,
                f'{self.name}.{self.sector_name}.capital_start': 273.1805902,  # 2019 value for test
                f'{self.name}.{self.sector_name}.sector_name': self.sector_name,
                f"{self.name}.{self.sector_name}.{'productivity_start'}": 1.31162,
                f"{self.name}.{self.sector_name}.{'capital_start'}": 100.92448579,
                f"{self.name}.{self.sector_name}.{'productivity_gr_start'}": 0.0027844,
                f"{self.name}.{self.sector_name}.{'decline_rate_tfp'}": 0.098585,
                f"{self.name}.{self.sector_name}.{'energy_eff_k'}": 0.1,
                f"{self.name}.{self.sector_name}.{'energy_eff_cst'}": 0.490463,
                f"{self.name}.{self.sector_name}.{'energy_eff_xzero'}": 1993,
                f"{self.name}.{self.sector_name}.{'energy_eff_max'}": 2.35832,
                f"{self.name}.{self.sector_name}.{'output_alpha'}": 0.99,
                f'{self.name}.{GlossaryCore.SectionList}': self.section_list,
                f"{self.name}.{self.sector_name}.{'depreciation_capital'}": 0.058,
                f'{self.name}.{GlossaryCore.EnergyMarketRatioAvailabilitiesValue}': self.energy_market_ratios,
                f'{self.name}.assumptions_dict': {
                    'compute_gdp': True,
                    'compute_climate_impact_on_gdp': True,
                    'activate_climate_effect_population': True,
                    'activate_pandemic_effects': True,
                },
                }
    def test_execute_sector_discipline(self):
        self.model_name = self.sector_name
