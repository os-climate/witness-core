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
import numpy as np
from scipy.optimize import minimize

from climateeconomics.calibration.crop.productions import (
    dict_of_production_in_megatons_2021,
)
from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore

# Road map calibration of crop discipline :

# We want the discipline to output a production for each food type, based on a capital.
# We have the total capital for agriculture in 2021, and we want to distribute it among the food types.
# We have the production in megatons for each food type in 2021. (from production calibration script)
# Chat GPT gave the following estimation for capex intensity for each food type which makes sense
# We want to find the capex per ton for each food type that will match the total capital for agriculture in 2021.
# We will use a loss function that will be the relative difference between the total capital modeled and the actual capital for agriculture in 2021.
# We will use the scipy minimize function to find the optimal capex per ton for each food type, in order to match the total capital for agriculture in 2021 AND production of each food type.
# Then we will be able to obtain the capital of each food type : production * deduced capex per ton
# We also have the investments for the agriculture sector in 2021, we will distribute it among the food types proportionally to the capital share of each food type, to deduce the investment for each food type.


capex_per_ton_value_rangechat_gpt = { # initial value, min value, max value
    GlossaryCore.RedMeat: (5250.0, 4000, 6500),
    GlossaryCore.Fish: (4250.0, 3000, 5500),
    GlossaryCore.WhiteMeat: (1750.0, 1000, 2500),
    GlossaryCore.Milk: (1500.0, 500, 1500),
    GlossaryCore.Eggs: (1150.0, 800, 1500),
    GlossaryCore.Rice: (450.0, 300, 600),
    GlossaryCore.Maize: (250.0, 150, 350),
    GlossaryCore.SugarCane: (350, 300, 500),
    GlossaryCore.Cereals: (600.0, 400, 800),
    GlossaryCore.FruitsAndVegetables: (1000, 500, 1500),
    GlossaryCore.OtherFood: (2500.0, 1000, 4000),
}

mapper_food_type_index = {food_type: i for i, food_type in enumerate(GlossaryCore.DefaultFoodTypesV2)}
mapper_index_food_type = {i: food_type for i, food_type in enumerate(GlossaryCore.DefaultFoodTypesV2)}
"""
constraints_greater_or_equal = [
    (GlossaryCore.RedMeat, GlossaryCore.WhiteMeat),
    (GlossaryCore.Fish, GlossaryCore.WhiteMeat),
    (GlossaryCore.WhiteMeat, GlossaryCore.RiceAndMaize),
    (GlossaryCore.WhiteMeat, GlossaryCore.Milk),
    (GlossaryCore.WhiteMeat, GlossaryCore.Cereals),
    (GlossaryCore.WhiteMeat, GlossaryCore.Eggs),
    (GlossaryCore.WhiteMeat, GlossaryCore.FruitsAndVegetables),
]
constraints_matrix = np.zeros((len(constraints_greater_or_equal), len(GlossaryCore.DefaultFoodTypesV2)))
for i, (food_type1, food_type2) in enumerate(constraints_greater_or_equal):
    constraints_matrix[i, mapper_food_type_index[food_type1]] = 1
    constraints_matrix[i, mapper_food_type_index[food_type2]] = -1
constraint_vector = np.zeros(len(constraints_greater_or_equal))
constraints = [LinearConstraint(constraints_matrix, lb=constraint_vector)] # 1% tolerance
"""


def loss_function(x: np.ndarray):
    actual_capital_agriculture_2021 = DatabaseWitnessCore.SectorAgricultureCapital2021.value
    total_capital_modeled = 0
    for i, food_type_design_value_capex_per_ton in enumerate(x):
        food_type = mapper_index_food_type[i]
        # ($/ton) * (Mt) = $ * 10^6 = M$
        # so divide by 1000 to get B$
        capital_dict_food_type = food_type_design_value_capex_per_ton * dict_of_production_in_megatons_2021[food_type] / 1000.
        total_capital_modeled += capital_dict_food_type / 1000 # divide by 1000 to get Trillion $
    loss = abs(((total_capital_modeled - actual_capital_agriculture_2021) / actual_capital_agriculture_2021))
    #print("loss", loss)
    return loss

x0 = np.array([capex_per_ton_value_rangechat_gpt[food_type][0] for food_type in GlossaryCore.DefaultFoodTypesV2])
bounds = [capex_per_ton_value_rangechat_gpt[food_type][1:] for food_type in GlossaryCore.DefaultFoodTypesV2]
result = minimize(loss_function, x0)#, constraints=constraints)
print("Relative error on total capital for agriculture in 2021:")
print(round(result.fun,2))
resulting_capex_per_ton = {mapper_index_food_type[i]: np.round(value) for i, value in enumerate(result.x)}
sorted(resulting_capex_per_ton.items(), key=lambda x: x[1])
#print("Optimal capex per ton to match total Agriculture capital ($/ton):")
#print(sorted(resulting_capex_per_ton.items(), key=lambda x: x[1], reverse=True))

capital_start_food_type_breakdown = {}
for food_type, capex_per_ton_optimal in resulting_capex_per_ton.items():
    capital_dict_food_type = capex_per_ton_optimal * dict_of_production_in_megatons_2021[food_type] / 1000.
    capital_start_food_type_breakdown[food_type] = np.round(capital_dict_food_type)

share_of_capital_sector_food_type = {food_type: np.round(capital_dict_food_type / 1000 / DatabaseWitnessCore.SectorAgricultureCapital2021.value * 100, 2) for food_type, capital_dict_food_type in capital_start_food_type_breakdown.items()}
#print("Agricultre capital breakdown by food type (%):")
#print(sorted(share_of_capital_sector_food_type.items(), key=lambda x: x[1], reverse=True))

#print("Agricultre capital start by food type (B$):")
#print(sorted(capital_start_food_type_breakdown.items(), key=lambda x: x[1], reverse=True))

# capital * capital intensity = production => capital intensity = production (Mt) / capital (B$)
# Mt / B$ = 10^6 / 10^9 = 10^-3 t / $ = kg / $ = t / k$
capital_intensity_food_types = {food_type: np.round(dict_of_production_in_megatons_2021[food_type] / capital_start_food_type_breakdown[food_type], 2) for food_type in GlossaryCore.DefaultFoodTypesV2}

#print("Capital intensity by food types (t / k$):")
##print(sorted(capital_intensity_food_types.items(), key=lambda x: x[1], reverse=True))

invest_food_type_share_start = {food_type: share_of_capital_sector_food_type[food_type] for food_type in GlossaryCore.DefaultFoodTypesV2}
invest_food_type_start = {food_type: np.round(invest_food_type_share_start[food_type] * DatabaseWitnessCore.SectorAgricultureInvest2021.value * 1000, 2) for food_type in GlossaryCore.DefaultFoodTypesV2} # in billion $
# Save the dictionaries to a JSON file
to_export = {
    "capital_start_food_type": capital_start_food_type_breakdown,
    "capital_intensity_food_type": capital_intensity_food_types,
    "invest_food_type_share_start": share_of_capital_sector_food_type, # at year start we invest in each food type proportionally to the capital share
    "invest_food_type_start": invest_food_type_start,
}