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
from copy import deepcopy

import numpy as np
import pandas as pd
from energy_models.core.stream_type.resources_models.resource_glossary import (
    ResourceGlossary,
)
from sostrades_core.tools.base_functions.exp_min import (
    compute_dfunc_with_exp_min,
    compute_func_with_exp_min,
)

from climateeconomics.core.core_resources.resource_model.resource_model import (
    ResourceModel,
)
from climateeconomics.glossarycore import GlossaryCore


class OrderOfMagnitude():
    KILO = 'k'
    # USD_PER_USton = 'USD/USton'
    # MILLION_TONNES='million_tonnes'

    magnitude_factor = {
        KILO: 10 ** 3
        # USD_PER_USton:1/0.907
        # MILLION_TONNES: 10**6
    }


class CopperResourceModel(ResourceModel):
    """
    Resource pyworld3
    General implementation of a resource pyworld3, to be inherited by specific models for each type of resource
    """

    resource_name = ResourceGlossary.Copper['name']

    def configure_parameters(self, inputs_dict):
        super().configure_parameters(inputs_dict)
        self.sectorisation_dict = inputs_dict['sectorisation']
        self.resource_max_price = inputs_dict['resource_max_price']

    # Units conversion
    oil_barrel_to_tonnes = 6.84
    bcm_to_Mt = 1 / 1.379
    kU_to_Mt = 10 ** -6

    # To convert from 1E6 oil_barrel to Mt
    conversion_factor = 1.0
    # Average copper price rise and decrease
    price_rise = 1.494
    price_decrease = 0.95

    def convert_demand(self, demand):
        self.resource_demand = demand
        self.resource_demand[self.resource_name] = demand[self.resource_name]

    def get_global_demand(self, demand):
        energy_demand = self.sectorisation_dict['power_generation']
        self.resource_demand = demand
        self.resource_demand[self.resource_name] = demand[self.resource_name] / energy_demand
        self.conversion_factor = 1 / energy_demand

    def sigmoid(self, ratio, sigmoid_min):
            '''
            function sigmoid redefined for the ratio :
            sigmoid([-10;10])->[0;1] becomes sigmoid([1;0])->[sigmoid_min, resource_max_price] (ratio = 1 matches sigmoids' lower bound, and ratio = 0 matches sigmoid's upper bound)
            '''
            x = -20 * ratio + 10

            sig = 1 / (1 + math.exp(-x))

            return sig * (self.resource_max_price - sigmoid_min) + sigmoid_min

    def compute_price(self):
        """
        price function depends on the ratio use_stock/demand
        price(ratio) = (price_max - price_min)(1-ratio) + price_min
        """

        # dataframe initialization
        self.resource_price['price'] = np.insert(np.zeros(len(self.years) - 1), 0, self.resource_price_data.loc[0, 'price'])
        resource_price_dict = self.resource_price.to_dict()
        self.resource_demand = self.resources_demand[[GlossaryCore.Years, self.resource_name]]
        self.get_global_demand(self.resource_demand)

        demand = deepcopy(self.resource_demand[self.resource_name].values)
        demand_limited = compute_func_with_exp_min(
            np.array(demand), 1.0e-10)

        self.ratio_usable_demand = np.maximum(self.use_stock[self.sub_resource_list[0]].values / demand_limited, 1E-15)

        for year_cost in self.years[1:]:

            resource_price_dict['price'][year_cost] = \
                    (self.resource_max_price - self.resource_price_data.loc[0, 'price']) *\
                        (1 - self.ratio_usable_demand[year_cost - self.year_start]) + self.resource_price_data.loc[0, 'price']

        self.resource_price = pd.DataFrame.from_dict(resource_price_dict)

    def get_d_price_d_demand(self, year_start, year_end, nb_years, grad_use, grad_price):
        ascending_price_resource_list = list(
            self.resource_price_data.sort_values(by=['price'])['resource_type'])

        demand = deepcopy(self.resource_demand[self.resource_name].values)
        demand_limited = compute_func_with_exp_min(np.array(demand), 1.0e-10)
        grad_demand_limited = compute_dfunc_with_exp_min(np.array(demand), 1.0e-10)

        self.ratio_usable_demand = np.maximum(self.use_stock[self.sub_resource_list[0]].values / demand_limited, 1E-15)
        # # ------------------------------------------------
        # # price is cst *u/v function with u = use and v = demand
        # # price gradient is cst * (u'v - uv') / v^2
        for year_demand in range(year_start + 1, year_end + 1):
            for resource_type in ascending_price_resource_list:

                for year in range(year_start + 1, year_demand + 1):
                    # grad_price = cst * u'v  / v^2 (cst < 0)
                    if self.use_stock[self.sub_resource_list[0]][year_demand] / demand_limited[year_demand - year_start] > 1E-15:
                        grad_price[year_demand - year_start, year - year_start] = \
                            - grad_use[resource_type][year_demand - year_start, year - year_start]
                        # grad_price -= cst *  uv'  / v^2 (cst < 0)
                        if year == year_demand:
                                grad_price[year_demand - year_start, year - year_start] += self.use_stock[self.sub_resource_list[0]][year_demand]\
                                        * self.conversion_factor / demand_limited[year_demand - year_start]
                        grad_price[year_demand - year_start, year - year_start] = grad_price[year_demand - year_start, year - year_start] *\
                            (self.resource_max_price - self.resource_price_data.loc[0, 'price']) *\
                            grad_demand_limited[year_demand - year_start] / demand_limited[year_demand - year_start]

        return grad_price
