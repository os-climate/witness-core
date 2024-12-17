'''
Copyright 2024 Capgemini

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
from copy import copy

from climateeconomics.calibration.crop.other_data import (
    energy_consumption_agri_2021,
    forestry_gdp_ppp_2021,
)
from climateeconomics.calibration.crop.productions import (
    dict_of_production_delivered_sold_2021,
    dict_of_raw_production_in_megatons_2021,
)

#https://www.statista.com/statistics/675826/average-prices-meat-beef-worldwide/
from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore

#https://fred.stlouisfed.org/series/PFISHUSDM


############## READ ME ##############
# In this script, we calibrate the prices and their composition for each food type produced in the Crop model.
# As the GDP of the Agriculture will be computed as GPD = (Sum over food types) Quantity food types X Price food type
# And given that we already have the quantities, we need to find the prices that will get us back on GDP.
# Remark : We need to exclude the revenus from forestry activities from the agriculture GDP.

agri_gdp_ppp_2021 = DatabaseWitnessCore.MacroInitGrossOutput.get_value_at_year(2021) * DatabaseWitnessCore.ShareGlobalGDPAgriculture2021.value /100
agri_gdp_ppp_2021_except_forestry = agri_gdp_ppp_2021 - forestry_gdp_ppp_2021
print("agriculture gdp 2021", agri_gdp_ppp_2021, "T$")
print("forestry gdp 2021", forestry_gdp_ppp_2021, "T$")
print("gdp agriculture except forestry 2021", agri_gdp_ppp_2021_except_forestry, "T$")

# https://www.statista.com/statistics/502286/global-meat-and-seafood-market-value/
# meat market value 2021 : 0.897 T$
# pork meat : 0.27990 T$ in 2023 https://www.precedenceresearch.com/pork-meat-market#:~:text=The%20global%20pork%20meat%20market,3.92%25%20from%202024%20to%202033.
# beef meat : 0.43660 T$ in 2023 https://www.fortunebusinessinsights.com/beef-market-106640#:~:text=Beef%20Market%20Size%20Overview&text=The%20global%20beef%20market%20size,5.52%25%20during%20the%20forecast%20period.

market_values = {
    GlossaryCore.Fish: 0.508, # grahpreader for 2021 T$ https://www.statista.com/outlook/cmo/food/fish-seafood/worldwide
    # Fruits  2021  https://www.grandviewresearch.com/industry-analysis/fresh-fruits-market-report#:~:text=The%20global%20fresh%20fruits%20market%20size%20was%20estimated%20at%20USD,USD%20708.1%20billion%20by%202028.
    # Vegetables 2021 https://www.grandviewresearch.com/industry-analysis/fresh-vegetables-market-report#:~:text=The%20global%20fresh%20vegetables%20market,2.8%25%20from%202022%20to%202028.
    GlossaryCore.FruitsAndVegetables: 0.5511 + 0.63254,

    GlossaryCore.Rice: 0.28745,  # Rice 2021 https://www.google.com/search?q=rice+market+value+global+2021&sca_esv=9a5e7dbe56d12357&rlz=1C1UEAD_enFR1027FR1027&sxsrf=ADLYWIJUOQxDjeBLYspnyiNKLioJfWJtnw%3A1733758604396&ei=jA5XZ5jnF9KokdUP6cOBqAY&ved=0ahUKEwiY3faNgpuKAxVSVKQEHelhAGUQ4dUDCA8&uact=5&oq=rice+market+value+global+2021&gs_lp=Egxnd3Mtd2l6LXNlcnAiHXJpY2UgbWFya2V0IHZhbHVlIGdsb2JhbCAyMDIxMggQIRigARjDBDIIECEYoAEYwwRIyQxQsgRYyAdwAngBkAEAmAF1oAHuAqoBAzMuMbgBA8gBAPgBAZgCBqAC_ALCAgoQABiwAxjWBBhHwgIKECEYoAEYwwQYCpgDAIgGAZAGCJIHAzUuMaAHgQ8&sclient=gws-wiz-serp
    GlossaryCore.Maize: 0.079,  # Maize 2021 ChatGPT,
    GlossaryCore.Milk: 0.894, # 2021 https://www.statista.com/statistics/502280/global-dairy-market-value/
    GlossaryCore.SugarCane: 48.60 / 1e3, # https://www.futuremarketinsights.com/reports/cane-sugar-market#:~:text=2019%20to%202023%20Global%20Cane,USD%2055%2C744%20million%20in%202023.
    GlossaryCore.Eggs: 0.22739, # 2021 https://www.prnewswire.com/news-releases/global-egg-market-report-2021-covid-19-impact-and-recovery-to-2030-301232415.html
    # other cereals, assumed 3.95 % growth from 2021 to 2024 as expected for 2024-2029 in this article.
    # other cerals = all cereals (1.03 T$ in 2024) -  maize - rice
    GlossaryCore.Cereals: 1.03 / 1.0395 ** 3 - 0.28745 - 0.079,
    GlossaryCore.RedMeat: 0.43660,
    GlossaryCore.WhiteMeat: 0.27990 / 1.03 ** 2 + (0.897 - 0.43660 / 1.03 ** 2) # assume 3% percent growth between 2021 and 2023
}
rescaling_factor = 0.85

market_values = {key: rescaling_factor * value for key, value in market_values.items()}
total_markets_size = sum(market_values.values())
all_food_types_except_other = copy(GlossaryCore.DefaultFoodTypesV2)
all_food_types_except_other.remove(GlossaryCore.OtherFood)
assert set(all_food_types_except_other).issubset(set(market_values.keys())) # make sure i didnt forget any food type
print("market size of all food except Other",round(total_markets_size, 2), "T$")
other_market_size = agri_gdp_ppp_2021_except_forestry - total_markets_size

print("Other food category market size", round(other_market_size, 2), "T$")

market_values[GlossaryCore.OtherFood] = other_market_size
assert abs((sum(market_values.values()) - agri_gdp_ppp_2021_except_forestry) / agri_gdp_ppp_2021_except_forestry) < 0.01
market_values = dict(sorted(market_values.items(), key=lambda item: item[1], reverse=True))
print('\nMarket size after readjustment 2021 (T$):')
for key, val in market_values.items():
    print('\t',key.capitalize(),':', round(val,3), "T$")

assert abs((sum(market_values.values()) - agri_gdp_ppp_2021_except_forestry) / agri_gdp_ppp_2021_except_forestry) < 0.01

# $ / t = G $ / Gt = (1000 T$ / 1000 t = T$ * 1000 / t
unitary_prices = {ft: round(market_values[ft] / dict_of_production_delivered_sold_2021[ft] * 1000, 2) * 1000 for ft in GlossaryCore.DefaultFoodTypesV2}

print('\nProdutions sold to consumers in 2021 (Megatons):')
for key, val in dict_of_production_delivered_sold_2021.items():
    print('\t',key.capitalize(),':', round(val), "Mt")




unitary_prices = dict(sorted(unitary_prices.items(), key=lambda item: item[1], reverse=True))
print('\nFinal prices to match individual market values and total Agriculture GDP in 2021:')
for key, val in unitary_prices.items():
    print('\t',key.capitalize(),':', val, "$/ton")

# in T $ : Mt * $/t = M$ -> divide by 1e6
gdp_quantity_times_price = sum([dict_of_production_delivered_sold_2021[ft] * unitary_prices[ft] for ft in GlossaryCore.DefaultFoodTypesV2]) / 1e6
# assert less than 1% error of gdp reconstructed
assert abs( (gdp_quantity_times_price - agri_gdp_ppp_2021_except_forestry) / agri_gdp_ppp_2021_except_forestry ) < 0.005

margins = {
    GlossaryCore.RedMeat: 15,              # Livestock farmers retain a smaller share due to feed and labor costs.
    GlossaryCore.WhiteMeat: 10,            # Poultry farmers face competitive pricing and high production costs.
    GlossaryCore.Milk: 25,                 # Dairy farmers often retain a higher share compared to meat producers.
    GlossaryCore.Eggs: 30,                 # Egg producers have relatively low overhead costs, retaining a larger share.
    GlossaryCore.Rice: 50,                 # Rice farmers often retain a significant share due to bulk production.
    GlossaryCore.Maize: 55,                # Similar to rice, maize producers retain a substantial portion.
    GlossaryCore.Cereals: 40,              # Cultivators of processed cereals retain less due to processing costs.
    GlossaryCore.FruitsAndVegetables: 30,  # High perishability and middlemen reduce cultivator margins.
    GlossaryCore.Fish: 20,                 # Aquaculture farmers often face high operational costs.
    GlossaryCore.SugarCane: 35,            # Margins are moderate due to bulk production and processing deductions.
    GlossaryCore.OtherFood: 30             # Generic estimate for miscellaneous food categories.
}
# price composition according to chatGPT and seems logical
cost_breakdown_share = {
    GlossaryCore.RedMeat: {
        GlossaryCore.FoodTypeCapitalAmortizationCostName: 0.125,
        GlossaryCore.FoodTypeCapitalMaintenanceCostName: 0.05,
        GlossaryCore.FoodTypeFeedingCostsName: 0.55,
        GlossaryCore.FoodTypeLaborCostByProdUnitName: 0.15,
        GlossaryCore.FoodTypeFertilizationAndPesticidesCostsName: 0.0,  # Not applicable
        "EnergyCosts": 0.125,
    },
    GlossaryCore.WhiteMeat: {
        GlossaryCore.FoodTypeCapitalAmortizationCostName: 0.10,
        GlossaryCore.FoodTypeCapitalMaintenanceCostName: 0.05,
        GlossaryCore.FoodTypeFeedingCostsName: 0.65,
        GlossaryCore.FoodTypeLaborCostByProdUnitName: 0.1,
        GlossaryCore.FoodTypeFertilizationAndPesticidesCostsName: 0.0,  # Not applicable
        "EnergyCosts": 0.1,
    },
    GlossaryCore.Milk: {
        GlossaryCore.FoodTypeCapitalAmortizationCostName: 0.125,
        GlossaryCore.FoodTypeCapitalMaintenanceCostName: 0.05,
        GlossaryCore.FoodTypeFeedingCostsName: 0.55,
        GlossaryCore.FoodTypeLaborCostByProdUnitName: 0.175,
        GlossaryCore.FoodTypeFertilizationAndPesticidesCostsName: 0.0,  # Not applicable
        "EnergyCosts": 0.1,
    },
    GlossaryCore.Eggs: {
        GlossaryCore.FoodTypeCapitalAmortizationCostName: 0.125,
        GlossaryCore.FoodTypeCapitalMaintenanceCostName: 0.075,
        GlossaryCore.FoodTypeFeedingCostsName: 0.55,
        GlossaryCore.FoodTypeLaborCostByProdUnitName: 0.15,
        GlossaryCore.FoodTypeFertilizationAndPesticidesCostsName: 0.0,  # Not applicable
        "EnergyCosts": 0.1,
    },
    GlossaryCore.Rice: {
        GlossaryCore.FoodTypeCapitalAmortizationCostName: 0.125,
        GlossaryCore.FoodTypeCapitalMaintenanceCostName: 0.05,
        GlossaryCore.FoodTypeFeedingCostsName: 0.0,
        GlossaryCore.FoodTypeLaborCostByProdUnitName: 0.3,
        GlossaryCore.FoodTypeFertilizationAndPesticidesCostsName: 0.4,
        "EnergyCosts": 0.125,
    },
    GlossaryCore.Maize: {
        GlossaryCore.FoodTypeCapitalAmortizationCostName: 0.15,
        GlossaryCore.FoodTypeCapitalMaintenanceCostName: 0.075,
        GlossaryCore.FoodTypeFeedingCostsName: 0.0,
        GlossaryCore.FoodTypeLaborCostByProdUnitName: 0.2,
        GlossaryCore.FoodTypeFertilizationAndPesticidesCostsName: 0.45,
        "EnergyCosts": 0.125,
    },
    GlossaryCore.Cereals: {
        GlossaryCore.FoodTypeCapitalAmortizationCostName: 0.15,
        GlossaryCore.FoodTypeCapitalMaintenanceCostName: 0.1,
        GlossaryCore.FoodTypeFeedingCostsName: 0.0,
        GlossaryCore.FoodTypeLaborCostByProdUnitName: 0.175,
        GlossaryCore.FoodTypeFertilizationAndPesticidesCostsName: 0.45,
        "EnergyCosts": 0.125,
    },
    GlossaryCore.FruitsAndVegetables: {
        GlossaryCore.FoodTypeCapitalAmortizationCostName: 0.075,
        GlossaryCore.FoodTypeCapitalMaintenanceCostName: 0.05,
        GlossaryCore.FoodTypeFeedingCostsName: 0.0,
        GlossaryCore.FoodTypeLaborCostByProdUnitName: 0.45,
        GlossaryCore.FoodTypeFertilizationAndPesticidesCostsName: 0.3,
        "EnergyCosts": 0.125,
    },
    GlossaryCore.Fish: {
        GlossaryCore.FoodTypeCapitalAmortizationCostName: 0.175,
        GlossaryCore.FoodTypeCapitalMaintenanceCostName: 0.075,
        GlossaryCore.FoodTypeFeedingCostsName: 0.50,
        GlossaryCore.FoodTypeLaborCostByProdUnitName: 0.125,
        GlossaryCore.FoodTypeFertilizationAndPesticidesCostsName: 0.0,  # Not applicable
        "EnergyCosts": 0.125,
    },
    GlossaryCore.SugarCane: {
        GlossaryCore.FoodTypeCapitalAmortizationCostName: 0.075,
        GlossaryCore.FoodTypeCapitalMaintenanceCostName: 0.05,
        GlossaryCore.FoodTypeFeedingCostsName: 0.0,
        GlossaryCore.FoodTypeLaborCostByProdUnitName: 0.45,
        GlossaryCore.FoodTypeFertilizationAndPesticidesCostsName: 0.35,
        "EnergyCosts": 0.075,
    },
    GlossaryCore.OtherFood: {
        GlossaryCore.FoodTypeCapitalAmortizationCostName: 0.15,
        GlossaryCore.FoodTypeCapitalMaintenanceCostName: 0.1,
        GlossaryCore.FoodTypeFeedingCostsName: 0.40,  # Ingredients
        GlossaryCore.FoodTypeLaborCostByProdUnitName: 0.15,
        GlossaryCore.FoodTypeFertilizationAndPesticidesCostsName: 0.0,  # Not typically applicable
        "EnergyCosts": 0.2,
    },
}

# check all shares adds up to 100%
for ft, shares_dict in cost_breakdown_share.items():
    sum_shares = sum(shares_dict.values())
    test =  abs(sum_shares - 1) < 0.001
    assert test

keys_to_store = [
    GlossaryCore.FoodTypeCapitalAmortizationCostName,
    GlossaryCore.FoodTypeCapitalMaintenanceCostName,
    GlossaryCore.FoodTypeFeedingCostsName,
    GlossaryCore.FoodTypeLaborCostByProdUnitName,
    GlossaryCore.FoodTypeFertilizationAndPesticidesCostsName,
]
production_costs = {ft: unitary_prices[ft] * (1 - margins[ft] / 100) for ft in GlossaryCore.DefaultFoodTypesV2}
to_export = {}
for key in keys_to_store:
    to_export.update({
        key: {ft: round(cost_breakdown_share[ft][key] * production_costs[ft], 2) for ft in GlossaryCore.DefaultFoodTypesV2}
    })

# deducing how much energy each food type has consumed based on energy costs

total_energy_cons_per_ft = {ft: dict_of_raw_production_in_megatons_2021[ft] * production_costs[ft] * cost_breakdown_share[ft]["EnergyCosts"] for ft in GlossaryCore.DefaultFoodTypesV2}
share_cons_energy = {ft: total_energy_cons_per_ft[ft] / sum(total_energy_cons_per_ft.values()) for ft in GlossaryCore.DefaultFoodTypesV2}
consumption_GWh_per_ft = {ft: share_cons_energy[ft] * energy_consumption_agri_2021.value * 10 ** 6 for ft in GlossaryCore.DefaultFoodTypesV2}


# GWh / Mt = k MWh / Mt = kWh/ton
energy_intensity_by_prod_unit = {ft: round(consumption_GWh_per_ft[ft] / dict_of_raw_production_in_megatons_2021[ft], 2) for ft in GlossaryCore.DefaultFoodTypesV2}
energy_intensity_by_prod_unit = dict(sorted(energy_intensity_by_prod_unit.items(), key=lambda item: item[1], reverse=True))
print(f"\nEnergy intensity by food type ({GlossaryCore.FoodTypeEnergyIntensityByProdUnitVar['unit']})")
for key, val in energy_intensity_by_prod_unit.items():
    print("\t",key.capitalize(),":", val)


# to match final price of one ton, energy cost should be this for 2021 :
#$ / t
energy_cost_per_ton_2021 = {ft: production_costs[ft] * cost_breakdown_share[ft]["EnergyCosts"] for ft in GlossaryCore.DefaultFoodTypesV2}

# retro engineer the optimal energy price to fall back on unitary prices
# ($/t) / (kWh/t) = $/ kWh
optimal_energy_price = {ft: energy_cost_per_ton_2021[ft] /  energy_intensity_by_prod_unit[ft] * 1000 for ft in GlossaryCore.DefaultFoodTypesV2}
print("\nOptimal energy price ($/MWh)")
for key, val in optimal_energy_price.items():
    print("\t",key.capitalize(),":", val)
to_export.update({
    GlossaryCore.FoodTypeEnergyIntensityByProdUnitName: energy_intensity_by_prod_unit,
    GlossaryCore.FoodTypesPriceMarginShareName: margins
})
