'''
Copyright 2024 Capgemini
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
from sostrades_optimization_plugins.models.test_class import GenericDisciplinesTestClass

from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


class Crop2JacobianTestCase(GenericDisciplinesTestClass):

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'Test'
        self.override_dump_jacobian = False
        self.show_graphs = False
        self.jacobian_test = False
        self.pickle_directory = dirname(__file__)
        self.year_start = 2021
        self.year_end = GlossaryCore.YearEndDefaultTest
        self.years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1

        self.crop_productivity_reduction = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.CropProductivityReductionName: - np.linspace(0, 12/ 100, year_range) * 0.,  # fake
        })

        self.damage_fraction = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.DamageFractionOutput: np.linspace(0 /100., 12 / 100., year_range) * 0., # 2020 value
        })

        self.investments_food_types = pd.DataFrame({
            GlossaryCore.Years: self.years,  # 0.61 T$ (2020 value)
            **{food_type: DatabaseWitnessCore.SectorAgricultureInvest.get_value_at_year(2021) * GlossaryCore.crop_calibration_data['invest_food_type_share_start'][food_type] / 100. * 1000. for food_type in GlossaryCore.DefaultFoodTypesV2}  # convert to G$
        })
        self.workforce_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.SectorAgriculture: np.linspace(935., 935. * 1000, year_range),  # millions of people (2020 value)
        })
        population_2021 = 7_954_448_391
        self.population_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.PopulationValue: np.linspace(population_2021 / 1e6, 7870 * 1.2, year_range),  # millions of people (2021 value)
        })

        self.energy_market_ratios = pd.DataFrame({
            GlossaryCore.Years: self.years,
            "Total": 100.
        })

        self.energy_mean_price = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.EnergyPriceValue: 50.
        })

        self.ns_dict = {
            'ns_public': self.name,
            'ns_energy_market': self.name,
            GlossaryCore.NS_WITNESS: self.name,
            GlossaryCore.NS_CROP: f'{self.name}',
            'ns_sectors': f'{self.name}',
            GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
            GlossaryCore.NS_AGRI: f'{self.name}',
        }

    def get_inputs_dict(self) -> dict:
        return  {
            f'{self.name}.mdo_sectors_invest_level': 2,
            f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
            f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
            f'{self.name}.{GlossaryCore.CropProductivityReductionName}': self.crop_productivity_reduction,
            f'{self.name}.{GlossaryCore.WorkforceDfValue}': self.workforce_df,
            f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
            f'{self.name}.{GlossaryCore.EnergyMeanPriceValue}': self.energy_mean_price,
            f'{self.name}.{GlossaryCore.DamageFractionDfValue}': self.damage_fraction,
            f'{self.name}.{GlossaryCore.EnergyMarketRatioAvailabilitiesValue}': self.energy_market_ratios,
            f'{self.name}.Agriculture.{GlossaryCore.Crop}.{GlossaryCore.InvestmentDetailsDfValue}': self.investments_food_types
            }

    def test_crop_discipline_2(self):
        '''
        Check discipline setup and run
        '''
        self.model_name = 'Agriculture.Crop'
        self.mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.crop.crop_disc.CropDiscipline'
