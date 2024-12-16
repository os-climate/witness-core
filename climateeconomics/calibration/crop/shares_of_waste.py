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
from climateeconomics.calibration.crop.tools import (
    solve_share_consumers_waste,
    solve_share_prod_and_distrib_waste,
)
from climateeconomics.glossarycore import GlossaryCore

# Methodology:
# Most of the time FAO gives total production and total quantity wasted at supply chain
# -> we can deduce share of waste at supply chain
# Then if we are lucky we have the share of waste by consumers at home
# Otherwise we can find or assume a total waste end to end and deduce the share of waste by consumers at home

# FAO data food avalaible for consumption per pers per year
# https://ourworldindata.org/agricultural-production
food_waste_share_supply_chain = {
        #Fish waste at production, distribution, and by consumers.
        #total_waste = food_prod * (share_prod_waste / 100) + food_prod * (1 - share_prod_waste / 100) * share_consumers_waste / 100
        #share_consumers_waste = 2.043 / 23.813 * 100
        #share_prod_waste calculated using total_waste.
    GlossaryCore.Fish: solve_share_prod_and_distrib_waste(total_share_waste=14.78, share_waste_consumers=2043 / 23813 * 100),
    GlossaryCore.Rice: round(33.801 / 776.461 * 100, 2),
    GlossaryCore.Maize: round(62 / 1163 * 100, 2),
    GlossaryCore.SugarCane: round(120 / 1922 * 100, 2),
    GlossaryCore.Cereals: round(((63 / 1163 * 100) * 140.688 + (33 / 776 * 100) * 638.091) / (140.688 + 638.091), 2),
        #Waste supply chain for eggs.
        #share waste supply chain = waste supply chain / total production
    GlossaryCore.Eggs: round(4.710 / 93.297 * 100, 2),
        #Waste supply chain for vegetables and fruits.
        #(waste supply chain vegetables + fruits) / (total production vegetables + fruits)
    GlossaryCore.FruitsAndVegetables: round((112 + 74) / (1160 + 922) * 100, 2),
        #Took poultry as reference.
        #(waste supply chain) / (total production)
    GlossaryCore.WhiteMeat: round(0.654 / 136 * 100, 2),
        ## Took beef as reference.
        #(waste supply chain) / (total production)
    GlossaryCore.RedMeat: round(0.558 / 73.9 * 100, 2),
        #Waste supply chain for milk.
        #(waste supply chain) / (total production)
    GlossaryCore.Milk: round(26.187 / 940.6 * 100, 2),
    GlossaryCore.OtherFood: 2.5,  #Assumed, based on values for other food types.
}

# Most of the time there are numbers for the total waste of a certain food type (supply chain + consumers)
# But not directly by the consumer. For those case we will deduce it.
total_foodchain_wastes_w_supply_chain_waste_data = {
    GlossaryCore.Cereals: 30,  # https://openknowledge.fao.org/server/api/core/bitstreams/57f76ed9-6f19-4872-98b4-6e1c3e796213/content#:~:text=Studies%20commissioned%20by%20FAO%20estimated,and%2035%20percent%20of%20fish.
    GlossaryCore.Rice: 30,  # https://openknowledge.fao.org/server/api/core/bitstreams/57f76ed9-6f19-4872-98b4-6e1c3e796213/content#:~:text=Studies%20commissioned%20by%20FAO%20estimated,and%2035%20percent%20of%20fish.
    GlossaryCore.Maize: 30,  # https://openknowledge.fao.org/server/api/core/bitstreams/57f76ed9-6f19-4872-98b4-6e1c3e796213/content#:~:text=Studies%20commissioned%20by%20FAO%20estimated,and%2035%20percent%20of%20fish.
    GlossaryCore.Eggs: 7,
    GlossaryCore.Milk: 116 / 940 * 100,  # https://www.theguardian.com/environment/2018/nov/28/one-in-six-pints-of-milk-thrown-away-each-year-study-shows
    GlossaryCore.RedMeat: 53 / 318 * 100,  #https://viva.org.uk/planet/the-issues/food-waste/#:~:text=And%20of%20the%20263%20million,equivalent%20to%2075%20million%20cows.
    GlossaryCore.WhiteMeat: 53 / 318 * 100,  #https://viva.org.uk/planet/the-issues/food-waste/#:~:text=And%20of%20the%20263%20million,equivalent%20to%2075%20million%20cows.
}

# These are the value found online for share of waste by consumer at home
food_waste_share_consumers = {
    GlossaryCore.Fish: round(2043 / 23813 * 100, 2),  # https://www3.weforum.org/docs/WEF_Investigating_Global_Aquatic_Food_Loss_and_Waste_2024.pdf

    # weighted by production (FAO) average of fruits and vegetables share at home
    # https://www.toogoodtogo.com/en-au/about-food-waste;https://www.sciencedirect.com/science/article/pii/S0921344920302305#tbl0001
    GlossaryCore.FruitsAndVegetables: round(933 / (933 + 1173) * 12 + 1173 / (933 + 1173) * 25, 2),
}

for food_type, total_waste in total_foodchain_wastes_w_supply_chain_waste_data.items():
    food_waste_share_consumers[food_type] = solve_share_consumers_waste(total_share_waste=total_waste, share_waste_supply_chain=food_waste_share_supply_chain[food_type])

food_waste_share_consumers[GlossaryCore.SugarCane] = 10.0  # Assumed, based on values for other food types.
food_waste_share_consumers[GlossaryCore.OtherFood] = 10.0  # Assumed, based on values for other food types.

to_export = {
    GlossaryCore.FoodTypeWasteSupplyChainShareName: food_waste_share_supply_chain,
    GlossaryCore.FoodTypeWasteByConsumersShareName: food_waste_share_consumers
}

if __name__ == '__main__':
    print("\nWaste by supply chain share by food type:")
    for key, val in food_waste_share_supply_chain.items():
        print(f"{key}: {val['value']}")

    print("\nWaste by consumers share by food type:")
    for key, val in food_waste_share_consumers.items():
        print(f"{key}: {val}")

    print("\nTotal food waste by consumers share:")
    for food_type in GlossaryCore.DefaultFoodTypesV2:
        share_supply_chain = food_waste_share_supply_chain[food_type]['value']
        if food_type not in food_waste_share_consumers:
            print(f"Missing {GlossaryCore.FoodTypeWasteByConsumersShareName} data for {food_type}")
            continue
        else:
            share_consumers = food_waste_share_consumers[food_type]
            total_waste_share = round(100 - (1 - share_supply_chain / 100) * (1 - share_consumers / 100) * 100, 2)
            print(f"{food_type}: {total_waste_share}")

