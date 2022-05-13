'''
Copyright 2022 Airbus SAS

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
import numpy as np
import pandas as pd
from os.path import join, dirname
from copy import deepcopy

from climateeconomics.core.core_resources.resource_model.resource_model import ResourceModel
from energy_models.core.stream_type.resources_models.resource_glossary import ResourceGlossary
from climateeconomics.core.tools.Hubbert_Curve import compute_Hubbert_regression
from sos_trades_core.tools.base_functions.exp_min import compute_dfunc_with_exp_min,\
    compute_func_with_exp_min

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
    Resource model
    General implementation of a resource model, to be inherited by specific models for each type of resource
    """

    resource_name=ResourceGlossary.Copper['name']

    
    def configure_parameters(self, inputs_dict):
        super().configure_parameters(inputs_dict)
        self.sectorisation_dict = inputs_dict['sectorisation']
        self.resource_max_price = inputs_dict['resource_max_price']
        
    

    #Units conversion
    oil_barrel_to_tonnes = 6.84
    bcm_to_Mt = 1 / 1.379
    kU_to_Mt = 10 ** -6

    #To convert from 1E6 oil_barrel to Mt
    conversion_factor = 1
    #Average copper price rise and decrease
    price_rise = 1.494
    price_decrease = 0.95
    

    def convert_demand(self, demand):
        self.resource_demand=demand
        self.resource_demand[self.resource_name]=demand[self.resource_name]
    
    def get_global_demand(self, demand):
        energy_demand = self.sectorisation_dict['power_generation']
        self.resource_demand=demand
        self.resource_demand[self.resource_name]=demand[self.resource_name] * (100 / energy_demand)

    def sigmoid(self, ratio, sigmoid_min ):
            '''
            function sigmoid redefined for the ratio :
            sigmoid([-10;10])->[0;1] becomes sigmoid([1;0])->[sigmoid_min, resource_max_price] (ratio = 1 matches sigmoids' lower bound, and ratio = 0 matches sigmoid's upper bound)
            '''
            x= -20*ratio +10
            
            sig = 1/ (1 + math.exp(-x))
        
            # return sig*(self.resource_max_price-10057) + 10057
            return sig*(self.resource_max_price-sigmoid_min) + sigmoid_min
    
    def compute_price(self):
        
        # dataframe initialization
        self.resource_price = pd.DataFrame({'years': self.years})
        self.resource_price['price'] = np.insert(np.zeros(len(self.years)-1), 0, self.resource_price_data.loc[0, 'price'])
        resource_price_dict = self.resource_price.to_dict()
        self.resource_demand = self.resources_demand[['years', self.resource_name]]
        self.get_global_demand(self.resource_demand)

        available_resource = deepcopy(self.use_stock[self.sub_resource_list[0]].values[self.lifespan:])  
        available_resource_limited = compute_func_with_exp_min(
            np.array(available_resource), 1.0e-10)
      
        # Demand without ratio
        demand = deepcopy(self.resource_demand[self.resource_name].values)      
        demand_limited = compute_func_with_exp_min(
            np.array(demand), 1.0e-10)
        
        self.ratio_usable_demand = np.minimum(np.maximum(available_resource_limited / demand_limited, 1E-15), 1.0)
     
        for year_cost in self.years[1:] : 
            #if for 2 years straight the demand is too high the prices rise
            if self.ratio_usable_demand[year_cost - self.year_start] < 1 and self.ratio_usable_demand[year_cost - self.year_start -1] < 1 :
                resource_price_dict['price'][year_cost - resource_price_dict['years'][0]] = \
                    min(self.sigmoid(self.ratio_usable_demand[year_cost - self.year_start], resource_price_dict['price'][year_cost - 1- resource_price_dict['years'][0]] ), \
                         resource_price_dict['price'][year_cost -1 - resource_price_dict['years'][0]] * self.price_rise)
                    # max(self.sigmoid(self.ratio_usable_demand[year_cost - self.year_start], resource_price_dict['price'][year_cost - 1- resource_price_dict['years'][0]] ),\
                    #  resource_price_dict['price'][year_cost -1 - resource_price_dict['years'][0]])
            # if, after the prices rise, the production can answer the demand, the prices decrease, but less than they rose
            elif self.ratio_usable_demand[year_cost - self.year_start] == 1 and self.ratio_usable_demand[year_cost - self.year_start -1] == 1 and resource_price_dict['price'][year_cost -1 - resource_price_dict['years'][0]] != self.resource_price_data.loc[0, 'price']: 
                resource_price_dict['price'][year_cost - resource_price_dict['years'][0]] = \
                    max(resource_price_dict['price'][year_cost -1 - resource_price_dict['years'][0]] * self.price_decrease, resource_price_dict['price'][0])
            #if the price is at its minimum (initial value) and the demand is answered to, the prices stay the same
            else : 
                resource_price_dict['price'][year_cost - resource_price_dict['years'][0]] = \
                    resource_price_dict['price'][year_cost -1 - resource_price_dict['years'][0]]

        self.resource_price= pd.DataFrame.from_dict(resource_price_dict)

       
    def get_grad_price (self, year_start, year_end, nb_years, year_demand, grad_use, grad_price):
        # Turn dataframes into dict of np array for faster computation time
        resource_price_data_dict = self.resource_price_data.to_dict()
        total_consumption_dict = self.total_consumption.to_dict()
        use_stock_dict = self.use_stock.to_dict()   