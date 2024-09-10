'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2024/06/24 Copyright 2023 Capgemini

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
import math
from os.path import dirname, join

import numpy as np
import pandas as pd
from energy_models.core.stream_type.resources_models.resource_glossary import (
    ResourceGlossary,
)

from climateeconomics.glossarycore import GlossaryCore


def sigmoid(ratio):
    x = -20 * ratio + 10
    print(x)

    sig = 1 / (1 + math.exp(-x))

    return sig * 5 * 10057 + 10057

# p1 = sigmoid(0)
# p2 = sigmoid(1)
# p3 = sigmoid(0.5)
# print("les valeurs sont :  0 1 et 0.5")
# print([p1, p2, p3])


coal_name = ResourceGlossary.Coal['name']
copper_name = ResourceGlossary.Copper['name']

year_start = GlossaryCore.YearStartDefault
year_end = GlossaryCore.YearEndDefault
years = np.arange(year_start, year_end + 1)
lifespan = 30
new_stock = 47

print(1 / (1 + 2)**2)
print(1 / ((1 + 2)**2))

copper = pd.read_csv(join(dirname(__file__), f'../resources_data/{copper_name}_consumed_data.csv'))  # pd.read_csv(join(dirname(__file__),'copper_resource_consumed_data.csv')) ou : pd.DataFrame(columns= [GlossaryCore.Years , 'copper_consumption' ])
copper_production_data = pd.read_csv(join(dirname(__file__), f'../resources_data/{copper_name}_production_data.csv'))

copper_dict = copper.to_dict()
print("le dico")
print(copper_dict)
print("les cles du dico")
print(list(copper_dict['copper_consumption'].keys()))
# print("la conso là")
# print(copper_consumed_dict['copper_consumption'])

coal = pd.read_csv(join(dirname(__file__), f'../resources_data/{coal_name}_consumed_data.csv'))
coal_production_data = pd.read_csv(join(dirname(__file__), f'../resources_data/{coal_name}_production_data.csv'))

use_stock = pd.DataFrame(
            {GlossaryCore.Years: np.insert(years, 0, np.arange(year_start - lifespan, year_start, 1))})
# print(copper_dict['copper_consumption'])
# print(copper_['copper_consumption'] * 1000)

copper_sub_resource_list = [col for col in list(copper_production_data.columns) if col != GlossaryCore.Years]
copper_dict = {}
for resource_type in copper_sub_resource_list:
    use_stock[resource_type] = np.insert(np.zeros(len(years) - 1), 0, copper[f'{resource_type}_consumption'])
# print(use_stock['copper'])

coal_sub_resource_list = [col for col in list(coal_production_data.columns) if col != GlossaryCore.Years]
coal_dict = {}
for resource_type in coal_sub_resource_list:

    coal_dict[resource_type] = coal[f'{resource_type}_consumption'].values

# print (coal_dict)


"""copper_oui= copper.to_dict()

year = 2021
year_end = GlossaryCore.YearEndDefault + 1

years = np.arange(year, year_end)

copper_test = pd.DataFrame({GlossaryCore.Years: years , 'copper_consumption': np.linspace(0, 0, len(years))})

copper_new = pd.concat([copper, copper_test], ignore_index=True)
copper_dict = copper_new.to_dict()

# print("copper dataframe: \n")
# print(copper)
# print("copper dico : \n")
# print(copper_oui)





#test insert function

lifespan = 30
test = np.arange(1, 6, 1)
test = np.zeros(len(test))

fonction_test = np.insert(years, 0, np.arange(year - lifespan , year, 1))

fonction_test = np.insert(fonction_test, 0, test[:-1])

#print(test)

#print(copper_dict['copper_consumption'])


#visualisation

oui = copper['copper_consumption'].values

available_resource = deepcopy(oui[6:])

#print (available_resource)


#######

#regarder de plus près le dataframe usestock
# print(" copper consumed normal : \n")
# print(copper['copper_consumption'])
# print("\n")
# print("on insère au début unn truc férent\n")
# copper['copper_consumption'] = np.insert(copper['copper_consumption'], 0, resource_consumed_data['sub_bituminous_and_lignite_consumption'])
# print(copper['copper_consumption'])

"""
