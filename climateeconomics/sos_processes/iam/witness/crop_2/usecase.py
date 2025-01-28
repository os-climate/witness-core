'''
Copyright 2024 Capgemini
Modifications on 2023/04/19-2024/06/24 Copyright 2023 Capgemini

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
from typing import Union

import numpy as np
import pandas as pd
from sostrades_core.study_manager.study_manager import StudyManager

from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


class Study(StudyManager):
    def __init__(self, data: Union[None, dict]=None, execution_engine=None, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault):
        super().__init__(__file__, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.data = data

    def setup_usecase(self, study_folder_path=None):
        if self.data is not None:
            return self.data
        ns_study = self.ee.study_name
        model_name = 'AgricultureMix.Crop'
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1

        crop_productivity_reduction = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.CropProductivityReductionName: np.linspace(0., 4.5, len(years)) * 0,  # fake
        })

        damage_fraction = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.DamageFractionOutput: np.linspace(0.0043, 0.032, len(years)), # 2020 value
        })

        true_invests_agri_df = DatabaseWitnessCore.SectorAgricultureInvest.value
        true_invests_agri_df = true_invests_agri_df.loc[true_invests_agri_df[GlossaryCore.Years] >= self.year_start]
        last_year_invest = true_invests_agri_df[GlossaryCore.Years].max()
        n_missing_years = self.year_end - last_year_invest
        invest_agri = np.array(list(true_invests_agri_df['investment']) + [true_invests_agri_df['investment'].values[-1]] * n_missing_years)

        investments = pd.DataFrame({
            GlossaryCore.Years: years,
            **{food_type: invest_agri *
                          GlossaryCore.crop_calibration_data['invest_food_type_share_start'][food_type] / 100. * 1000.
               for food_type in GlossaryCore.DefaultFoodTypesV2}  # convert to G$
        })
        workforce_df = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.SectorAgriculture: np.linspace(935., 935*50, year_range)  # millions of people (2020 value)
        })

        population_2021 = 7_954_448_391
        population_df = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.PopulationValue: np.linspace(population_2021 / 1e6, 7870 * 1.2, year_range),
        })

        enegy_agri = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.TotalProductionValue: np.linspace(2591. /1000., 2591. /1000. * 50, year_range)  # PWh, 2020 value
        })

        energy_mean_price = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.EnergyPriceValue: np.linspace(70, 120, year_range)
        })


        inputs_dict = {
            f'{ns_study}.{GlossaryCore.YearStart}': self.year_start,
            f'{ns_study}.{GlossaryCore.YearEnd}': self.year_end,
            f'{ns_study}.{GlossaryCore.EnergyMeanPriceValue}': energy_mean_price,
            f'{ns_study}.{GlossaryCore.CropProductivityReductionName}': crop_productivity_reduction,
            f'{ns_study}.{GlossaryCore.WorkforceDfValue}': workforce_df,
            f'{ns_study}.{GlossaryCore.PopulationDfValue}': population_df,
            f'{ns_study}.{GlossaryCore.DamageFractionDfValue}': damage_fraction,
            f'{ns_study}.Macroeconomics.{GlossaryCore.SectorAgriculture}.{GlossaryCore.StreamProductionValue}': enegy_agri,
            f'{ns_study}.{model_name}.{GlossaryCore.FoodTypesInvestName}': investments,
        }

        return inputs_dict


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.test()
