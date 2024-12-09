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
#https://www.statista.com/statistics/675826/average-prices-meat-beef-worldwide/
from climateeconomics.database import DatabaseWitnessCore

#https://fred.stlouisfed.org/series/PFISHUSDM

agri_gdp_ppp_2021 = DatabaseWitnessCore.MacroInitGrossOutput.get_value_at_year(2021) * DatabaseWitnessCore.ShareGlobalGDPAgriculture2021.value /100
print(agri_gdp_ppp_2021)

# https://www.statista.com/statistics/502286/global-meat-and-seafood-market-value/
# meat market value 2021 : 0.897 T$


# price composition

cost_breakdown = {
    "RedMeat": {
        "CapitalAmortization": 0.125,
        "CapitalMaintenance": 0.05,
        "Feeding": 0.55,
        "EnergyCosts": 0.075,
        "Labor": 0.15,
        "FertilizationAndPesticides": 0.0,  # Not applicable
    },
    "WhiteMeat": {
        "CapitalAmortization": 0.10,
        "CapitalMaintenance": 0.05,
        "Feeding": 0.65,
        "EnergyCosts": 0.075,
        "Labor": 0.075,
        "FertilizationAndPesticides": 0.0,  # Not applicable
    },
    "Milk": {
        "CapitalAmortization": 0.125,
        "CapitalMaintenance": 0.05,
        "Feeding": 0.55,
        "EnergyCosts": 0.075,
        "Labor": 0.15,
        "FertilizationAndPesticides": 0.0,  # Not applicable
    },
    "Eggs": {
        "CapitalAmortization": 0.10,
        "CapitalMaintenance": 0.05,
        "Feeding": 0.55,
        "EnergyCosts": 0.125,
        "Labor": 0.075,
        "FertilizationAndPesticides": 0.0,  # Not applicable
    },
    "Rice": {
        "CapitalAmortization": 0.125,
        "CapitalMaintenance": 0.05,
        "Feeding": 0.0,
        "EnergyCosts": 0.125,
        "Labor": 0.25,
        "FertilizationAndPesticides": 0.35,
    },
    "Maize": {
        "CapitalAmortization": 0.125,
        "CapitalMaintenance": 0.05,
        "Feeding": 0.0,
        "EnergyCosts": 0.125,
        "Labor": 0.15,
        "FertilizationAndPesticides": 0.45,
    },
    "Cereals": {
        "CapitalAmortization": 0.125,
        "CapitalMaintenance": 0.05,
        "Feeding": 0.0,
        "EnergyCosts": 0.125,
        "Labor": 0.15,
        "FertilizationAndPesticides": 0.45,
    },
    "FruitsAndVegetables": {
        "CapitalAmortization": 0.075,
        "CapitalMaintenance": 0.05,
        "Feeding": 0.0,
        "EnergyCosts": 0.125,
        "Labor": 0.45,
        "FertilizationAndPesticides": 0.25,
    },
    "Fish": {
        "CapitalAmortization": 0.175,
        "CapitalMaintenance": 0.075,
        "Feeding": 0.50,
        "EnergyCosts": 0.125,
        "Labor": 0.125,
        "FertilizationAndPesticides": 0.0,  # Not applicable
    },
    "SugarCane": {
        "CapitalAmortization": 0.075,
        "CapitalMaintenance": 0.05,
        "Feeding": 0.0,
        "EnergyCosts": 0.075,
        "Labor": 0.45,
        "FertilizationAndPesticides": 0.35,
    },
    "OtherFood": {
        "CapitalAmortization": 0.075,
        "CapitalMaintenance": 0.05,
        "Feeding": 0.40,  # Ingredients
        "EnergyCosts": 0.175,
        "Labor": 0.15,
        "FertilizationAndPesticides": 0.0,  # Not typically applicable
    },
}
