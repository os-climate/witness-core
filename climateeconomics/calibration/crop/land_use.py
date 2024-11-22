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

from climateeconomics.calibration.crop.productions import dict_of_production_in_megatons_2021
from climateeconomics.glossarycore import GlossaryCore

# Methodology:
# We found land use values per kg of each food type
# but when multiplying by the production of each food type, we don't get the actual land use value from FAO report
# We will scale the land use values per kg of each food type to match the actual land use value from FAO report
# This way we conserve the relative order of land use values per kg of each food type which seemed very coherent


actual_crop_land_use_2021 = 4.8
land_use_by_ft = {
    # https://ourworldindata.org/land-use
    GlossaryCore.RedMeat: 345.,
    GlossaryCore.WhiteMeat: 14.5,
    GlossaryCore.Milk: 8.95,
    GlossaryCore.Eggs: 6.27,
    GlossaryCore.Fish: 0.,
    GlossaryCore.OtherFood: 5.1041,
    # using yield values of https://ourworldindata.org/agricultural-production :
    GlossaryCore.FruitsAndVegetables: round(10 / ((13.70 * 922 + 20.5 * 1160) / (922 + 1160)), 2), # weighted average of fruits and vegetables
    GlossaryCore.Rice: round(10 / 4.74, 2),
    GlossaryCore.Maize: round(10 / 5.87, 2),
    GlossaryCore.Cereals: round(10 / 4.15, 2),
    GlossaryCore.SugarCane: round(10 / 72.24, 2),
}
# Mt * m2 / kg = G m² => divide by 10^4 to get G ha
total_modeled_land_use = sum(dict_of_production_in_megatons_2021[ft] * land_use_by_ft[ft] / 1e4 for ft in GlossaryCore.DefaultFoodTypesV2)

factor_to_apply = actual_crop_land_use_2021 / total_modeled_land_use

land_use_by_ft = {ft: round(land_use_by_ft[ft] * factor_to_apply, 2) for ft in GlossaryCore.DefaultFoodTypesV2}
total_modeled_land_use = sum(dict_of_production_in_megatons_2021[ft] * land_use_by_ft[ft] / 1e4 for ft in GlossaryCore.DefaultFoodTypesV2)

print("Relative error on land use:", abs(actual_crop_land_use_2021 - total_modeled_land_use) / actual_crop_land_use_2021)

to_export = {
    GlossaryCore.FoodTypeLandUseByProdUnitName: land_use_by_ft,
}
