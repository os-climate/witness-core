"""
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
"""
from climateeconomics.calibration.crop.data import output_calibration_datas
from climateeconomics.glossarycore import GlossaryCore

capex_per_ton = {
    GlossaryCore.RedMeat: 5250.0,
    GlossaryCore.WhiteMeat: 1750.0,
    GlossaryCore.Milk: 1500.0,
    GlossaryCore.Eggs: 1150.0,
    GlossaryCore.RiceAndMaize: 500.0,
    GlossaryCore.Cereals: 600.0,
    GlossaryCore.FruitsAndVegetables: 1000,
    GlossaryCore.Fish: 4250.0,
    GlossaryCore.OtherFood: 2500.0
}

capex_per_ton_range_chat_gpt = {
    "RedMeat": (4000, 6500),
    "WhiteMeat": (1000, 2500),
    "Milk": (1000, 2000),
    "Eggs": (800, 1500),
    "RiceAndMaize": (300, 700),
    "Cereals": (400, 800),
    "FruitsAndVegetables": (500, 5000),
    "Fish": (3000, 5500),
    "OtherFood": (1000, 4000)
}


dict_of_production_in_megatons = {}
for data in output_calibration_datas:
    if data.varname == GlossaryCore.FoodTypeProductionName:
        dict_of_production_in_megatons[data.key] = data.value

actual_capital_agriculture_2022 = 6.5  # T$
total_capital = 0
capital_dict = {}
for food_type in GlossaryCore.DefaultFoodTypes:
    if food_type in capex_per_ton and food_type in dict_of_production_in_megatons:
        # ($/ton) * (Mt) = $ * 10^6 = M$
        # so divide by 1000 to get B$
        capital_dict[food_type] = capex_per_ton[food_type] * dict_of_production_in_megatons[food_type] / 1000.
        total_capital += capital_dict[food_type] / 1000 # divide by 1000 to get Trillion $

print(capital_dict)
print(total_capital)



