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
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.study_manager.study_manager import StudyManager

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
        model_name = 'Crop2'
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1

        crop_productivity_reduction = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.CropProductivityReductionName: 0.,  # fake
        })

        damage_fraction = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.DamageFractionOutput: 0.43, # 2020 value
        })

        investments = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.InvestmentsValue: 0.61, # T$ (2020 value)
        })
        workforce_df = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.SectorAgriculture: 935.,  # millions of people (2020 value)
        })

        population_df = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.PopulationValue: np.linspace(7870, 9000, year_range),  # millions of people (2020 value)
        })

        enegy_agri = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.TotalProductionValue: 2591. /1000.,  # PWh, 2020 value
        })

        dict_inputs = {
            GlossaryCore.FoodTypeEmissionsByProdUnitName.format(GlossaryCore.CO2): {
                GlossaryCore.RedMeat: 0.0,
                GlossaryCore.WhiteMeat: 3.95,
                GlossaryCore.Milk: 0.0,
                GlossaryCore.Eggs: 1.88,
                GlossaryCore.RiceAndMaize: 0.84,
                GlossaryCore.Cereals: 0.12,
                GlossaryCore.FruitsAndVegetables: 0.44,
                GlossaryCore.Fish: 2.37,
                GlossaryCore.OtherFood: 0.48
            },
            GlossaryCore.FoodTypeEmissionsByProdUnitName.format(GlossaryCore.CH4): {
                GlossaryCore.RedMeat: 6.823e-1,
                GlossaryCore.WhiteMeat: 1.25e-2,
                GlossaryCore.Milk: 3.58e-2,
                GlossaryCore.Eggs: 0.0,
                GlossaryCore.RiceAndMaize: 3.17e-2,
                # negligible methane in this category
                GlossaryCore.Cereals: 0.0,
                GlossaryCore.FruitsAndVegetables: 0.0,
                # consider fish farm only
                GlossaryCore.Fish: 3.39e-2,
                GlossaryCore.OtherFood: 0.,
            },
            GlossaryCore.FoodTypeEmissionsByProdUnitName.format(GlossaryCore.N2O): {
                GlossaryCore.RedMeat: 9.268e-3,
                GlossaryCore.WhiteMeat: 3.90e-4,
                GlossaryCore.Milk: 2.40e-4,
                GlossaryCore.Eggs: 1.68e-4,
                GlossaryCore.RiceAndMaize: 9.486e-4,
                GlossaryCore.Cereals: 1.477e-3,
                GlossaryCore.FruitsAndVegetables: 2.63e-4,
                GlossaryCore.Fish: 0.,  # no crop or livestock related
                GlossaryCore.OtherFood: 1.68e-3,
            },
            GlossaryCore.FoodTypeKcalByProdUnitName: {
                GlossaryCore.RedMeat: 1551.05,
                GlossaryCore.WhiteMeat: 2131.99,
                GlossaryCore.Milk: 921.76,
                GlossaryCore.Eggs: 1425.07,
                GlossaryCore.RiceAndMaize: 2572.46,
                GlossaryCore.Cereals: 2964.99,
                GlossaryCore.FruitsAndVegetables: 559.65,
                GlossaryCore.Fish: 609.17,
                GlossaryCore.OtherFood: 3061.06,
            },
            GlossaryCore.FoodTypeLandUseByProdUnitName: {
                GlossaryCore.RedMeat: 345.,
                GlossaryCore.WhiteMeat: 14.5,
                GlossaryCore.Milk: 8.95,
                GlossaryCore.Eggs: 6.27,
                GlossaryCore.RiceAndMaize: 2.89,
                GlossaryCore.Cereals: 4.5,
                GlossaryCore.FruitsAndVegetables: 0.8,
                GlossaryCore.Fish: 0.,
                GlossaryCore.OtherFood: 5.1041,
            },
            GlossaryCore.FoodTypeEnergyNeedName: {
                GlossaryCore.RedMeat: 0.0001,
                GlossaryCore.WhiteMeat: 0.0001,
                GlossaryCore.Milk: 0.0001,
                GlossaryCore.Eggs: 0.0001,
                GlossaryCore.RiceAndMaize: 0.0001,
                GlossaryCore.Cereals: 0.0001,
                GlossaryCore.FruitsAndVegetables: 0.0001,
                GlossaryCore.Fish: 0.0001,
                GlossaryCore.OtherFood: 0.0001,
            },
            GlossaryCore.FoodTypeWorkforceNeedName: {
                GlossaryCore.RedMeat: 0.0001,
                GlossaryCore.WhiteMeat: 0.0001,
                GlossaryCore.Milk: 0.0001,
                GlossaryCore.Eggs: 0.0001,
                GlossaryCore.RiceAndMaize: 0.0001,
                GlossaryCore.Cereals: 0.0001,
                GlossaryCore.FruitsAndVegetables: 0.0001,
                GlossaryCore.Fish: 0.0001,
                GlossaryCore.OtherFood: 0.0001,
            },
            GlossaryCore.FoodTypeCapexName: {  # $ / ton
                GlossaryCore.RedMeat: 225.0,  # Average for red meat
                GlossaryCore.WhiteMeat: 150.0,  # Average for white meat
                GlossaryCore.Milk: 75.0,  # Average for milk
                GlossaryCore.Eggs: 113.5,  # Average for eggs
                GlossaryCore.RiceAndMaize: 31.75,  # Average for rice and maize
                GlossaryCore.Cereals: 37.5,  # Average for cereals
                GlossaryCore.FruitsAndVegetables: 84.9,  # Average for fruits and vegetables
                GlossaryCore.Fish: 300.0,  # Average for fish
                GlossaryCore.OtherFood: 100.0,  # General estimate for other food types
            },
        }
        dict_to_dataframes = {
            GlossaryCore.FoodTypeWasteAtProductionShareName: {
                GlossaryCore.RedMeat: 3,
                GlossaryCore.WhiteMeat: 3,
                GlossaryCore.Milk: 8,
                GlossaryCore.Eggs: 7,
                GlossaryCore.RiceAndMaize: 8,
                GlossaryCore.Cereals: 10,
                GlossaryCore.FruitsAndVegetables: 15,
                GlossaryCore.Fish: 10,
                GlossaryCore.OtherFood: 5,
            },
            GlossaryCore.FoodTypeWasteByConsumersShareName: {
                GlossaryCore.RedMeat: 3,
                GlossaryCore.WhiteMeat: 3,
                GlossaryCore.Milk: 8,
                GlossaryCore.Eggs: 7,
                GlossaryCore.RiceAndMaize: 8,
                GlossaryCore.Cereals: 10,
                GlossaryCore.FruitsAndVegetables: 15,
                GlossaryCore.Fish: 10,
                GlossaryCore.OtherFood: 5,
            },
            GlossaryCore.ShareInvestFoodTypesName: {
                GlossaryCore.RedMeat: 1 / 9 * 100.,
                GlossaryCore.WhiteMeat: 1 / 9 * 100.,
                GlossaryCore.Milk: 1 / 9 * 100.,
                GlossaryCore.Eggs: 1 / 9 * 100.,
                GlossaryCore.RiceAndMaize: 1 / 9 * 100.,
                GlossaryCore.Cereals: 1 / 9 * 100.,
                GlossaryCore.FruitsAndVegetables: 1 / 9 * 100.,
                GlossaryCore.Fish: 1 / 9 * 100.,
                GlossaryCore.OtherFood: 1 / 9 * 100.,
            },
            GlossaryCore.ShareEnergyUsageFoodTypesName: {
                GlossaryCore.RedMeat: 1 / 9 * 100.,
                GlossaryCore.WhiteMeat: 1 / 9 * 100.,
                GlossaryCore.Milk: 1 / 9 * 100.,
                GlossaryCore.Eggs: 1 / 9 * 100.,
                GlossaryCore.RiceAndMaize: 1 / 9 * 100.,
                GlossaryCore.Cereals: 1 / 9 * 100.,
                GlossaryCore.FruitsAndVegetables: 1 / 9 * 100.,
                GlossaryCore.Fish: 1 / 9 * 100.,
                GlossaryCore.OtherFood: 1 / 9 * 100.,
            },
            GlossaryCore.ShareWorkforceFoodTypesName: {
                GlossaryCore.RedMeat: 1 / 9 * 100.,
                GlossaryCore.WhiteMeat: 1 / 9 * 100.,
                GlossaryCore.Milk: 1 / 9 * 100.,
                GlossaryCore.Eggs: 1 / 9 * 100.,
                GlossaryCore.RiceAndMaize: 1 / 9 * 100.,
                GlossaryCore.Cereals: 1 / 9 * 100.,
                GlossaryCore.FruitsAndVegetables: 1 / 9 * 100.,
                GlossaryCore.Fish: 1 / 9 * 100.,
                GlossaryCore.OtherFood: 1 / 9 * 100.,
            },
            GlossaryCore.FoodTypeShareDedicatedToStreamProdName.format(GlossaryEnergy.biomass_dry): {
                GlossaryCore.RedMeat: 0.,
                GlossaryCore.WhiteMeat: 0.,
                GlossaryCore.Milk: 0.,
                GlossaryCore.Eggs: 0.,
                GlossaryCore.RiceAndMaize: 3,
                GlossaryCore.Cereals: 10,
                GlossaryCore.FruitsAndVegetables: 0.,
                GlossaryCore.Fish: 0.,
                GlossaryCore.OtherFood: 0.,
            },
            GlossaryCore.FoodTypeShareWasteBeforeDistribUsedToStreamProdName.format(GlossaryEnergy.biomass_dry): {
                GlossaryCore.RedMeat: 0.,
                GlossaryCore.WhiteMeat: 0.,
                GlossaryCore.Milk: 0.,
                GlossaryCore.Eggs: 0.,
                GlossaryCore.RiceAndMaize: 20.,
                GlossaryCore.Cereals: 30.,
                GlossaryCore.FruitsAndVegetables: 0.,
                GlossaryCore.Fish: 0.,
                GlossaryCore.OtherFood: 0.,
            },
            GlossaryCore.FoodTypeShareUserWasteUsedToStreamProdName.format(GlossaryEnergy.biomass_dry): {
                GlossaryCore.RedMeat: 0.,
                GlossaryCore.WhiteMeat: 0.,
                GlossaryCore.Milk: 0.,
                GlossaryCore.Eggs: 0.,
                GlossaryCore.RiceAndMaize: 10.,
                GlossaryCore.Cereals: 10.,
                GlossaryCore.FruitsAndVegetables: 0.,
                GlossaryCore.Fish: 0.,
                GlossaryCore.OtherFood: 0.,
            },
            GlossaryCore.FoodTypeShareDedicatedToStreamProdName.format(GlossaryEnergy.wet_biomass): {
                GlossaryCore.RedMeat: 0.,
                GlossaryCore.WhiteMeat: 0.,
                GlossaryCore.Milk: 0.,
                GlossaryCore.Eggs: 0.,
                GlossaryCore.RiceAndMaize: 0.,
                GlossaryCore.Cereals: 0.,
                GlossaryCore.FruitsAndVegetables: 5,
                GlossaryCore.Fish: 0.,
                GlossaryCore.OtherFood: 1.,
            },
            GlossaryCore.FoodTypeShareWasteBeforeDistribUsedToStreamProdName.format(GlossaryEnergy.wet_biomass): {
                GlossaryCore.RedMeat: 0.,
                GlossaryCore.WhiteMeat: 0.,
                GlossaryCore.Milk: 0.,
                GlossaryCore.Eggs: 0.,
                GlossaryCore.RiceAndMaize: 0.,
                GlossaryCore.Cereals: 0.,
                GlossaryCore.FruitsAndVegetables: 0.,
                GlossaryCore.Fish: 0.,
                GlossaryCore.OtherFood: 20.,
            },
            GlossaryCore.FoodTypeShareUserWasteUsedToStreamProdName.format(GlossaryEnergy.wet_biomass): {
                GlossaryCore.RedMeat: 0.,
                GlossaryCore.WhiteMeat: 0.,
                GlossaryCore.Milk: 0.,
                GlossaryCore.Eggs: 0.,
                GlossaryCore.RiceAndMaize: 10.,
                GlossaryCore.Cereals: 0.,
                GlossaryCore.FruitsAndVegetables: 0.,
                GlossaryCore.Fish: 0.,
                GlossaryCore.OtherFood: 5.,
            },
        }
        food_types = list(list(dict_inputs.values())[0].keys())
        inputs_dict = {
            f'{ns_study}.{GlossaryCore.YearStart}': self.year_start,
            f'{ns_study}.{GlossaryCore.YearEnd}': self.year_end,
            f'{ns_study}.{GlossaryCore.CropProductivityReductionName}': crop_productivity_reduction,
            f'{ns_study}.{GlossaryCore.WorkforceDfValue}': workforce_df,
            f'{ns_study}.{GlossaryCore.PopulationDfValue}': population_df,
            f'{ns_study}.{GlossaryCore.DamageFractionDfValue}': damage_fraction,
            f'{ns_study}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.EnergyProductionValue}': enegy_agri,
            f'{ns_study}.{model_name}.{GlossaryCore.FoodTypesName}': food_types,
            f'{ns_study}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.InvestmentDfValue}': investments,
        }
        for varname, default_dict_values_var in dict_to_dataframes.items():
            df = pd.DataFrame({
                GlossaryCore.Years: years,
                **default_dict_values_var
            })
            inputs_dict.update({f'{ns_study}.{model_name}.{varname}': df})

        inputs_dict.update({f'{ns_study}.{model_name}.{varname}': value for varname, value in dict_inputs.items()})
        return inputs_dict


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()
