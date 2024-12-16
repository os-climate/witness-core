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
from climateeconomics.calibration.agriculture_economy.prices import to_export
from climateeconomics.calibration.crop.other_data import workforce_agri_2021
from climateeconomics.calibration.crop.productions import (
    dict_of_production_delivered_sold_2021,
)
from climateeconomics.glossarycore import GlossaryCore

labor_costs_per_unit_prod = to_export[GlossaryCore.FoodTypeLaborCostByProdUnitName]
total_labor_cost_per_food_type = {ft: dict_of_production_delivered_sold_2021[ft] * labor_costs_per_unit_prod[ft] for ft in GlossaryCore.DefaultFoodTypesV2}

total_labor_cost = sum(total_labor_cost_per_food_type.values())
shares_labor_distribution = {ft: round(total_labor_cost_per_food_type[ft] / total_labor_cost, 2) for ft in GlossaryCore.DefaultFoodTypesV2}

for ft, share in shares_labor_distribution.items():
    print(ft.capitalize(), share)

print(sum(shares_labor_distribution.values()))

worker_per_ft = {ft: workforce_agri_2021.value * shares_labor_distribution[ft] for ft in GlossaryCore.DefaultFoodTypesV2}

workforce_fisheries_2022 = 61.8 # Million workers https://www.fao.org/newsroom/detail/fao-report-global-fisheries-and-aquaculture-production-reaches-a-new-record-high/en=


#### Energy

energy_cost_per_prod_unit = total_labor_cost_per_food_type

print(worker_per_ft)





