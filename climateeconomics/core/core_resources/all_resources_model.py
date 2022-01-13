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

import numpy as np
import pandas as pd
from copy import deepcopy
import os

class OrderOfMagnitude():
    KILO = 'k'
    #USD_PER_USton = 'USD/USton'
    #MILLION_TONNES='million_tonnes'

    magnitude_factor = {
        KILO: 10 ** 3
        # USD_PER_USton:1/0.907
        #MILLION_TONNES: 10**6
    }

class AllResourceModel():
    """
    All Resource model
    """
    YEAR_START = 'year_start'
    YEAR_END = 'year_end'
    ALL_RESOURCE_DEMAND = 'all_resource_demand'
    ALL_RESOURCE_STOCK = 'all_resource_stock'
    ALL_RESOURCE_PRICE = 'all_resource_price'
    All_RESOURCE_USE = 'all_resource_use'
    ALL_RESOURCE_PRODUCTION = 'all_resource_production'
    RATIO_USABLE_DEMAND='all_resource_ratio_usable_demand'
    ALL_RESOURCE_RATIO_PROD_DEMAND = 'all_resource_ratio_prod_demand'
    NON_MODELED_RESOURCE_PRICE = 'non_modeled_resource_price'
    RESOURCE_LIST=['oil_resource','uranium_resource','natural_gas_resource','coal_resource']
    
    def __init__(self, param):
        '''
        Constructor
        '''
        self.param = param
        self.all_resource_stock= None
        self.all_resource_price = None
        self.all_resource_use = None
        self.all_resource_production = None
        self.all_resource_ratio_prod_demand= None
        self.all_resource_ratio_usable_demand= None
        self.resource_demand=None
        self.set_data()
        self.price_gas_conversion=1.379*35310700*10**-6
        self.price_uranium_conversion=(1/0.001102)*0.907185
        self.price_oil_conversion=7.33
        self.price_coal_conversion=0.907185
        self.gas_conversion=1/1.379
        self.uranium_conversion=10**-6



    def set_data(self):
        self.year_start = self.param[AllResourceModel.YEAR_START]
        self.year_end = self.param[AllResourceModel.YEAR_END]

    def compute(self, input_dict):

        demand_df = deepcopy(input_dict['All_Demand'])
        demand_df.index=demand_df['years'].values
        # initialisation of dataframe:
        self.all_resource_stock = pd.DataFrame({'years': demand_df.index})
        self.all_resource_stock.set_index('years', inplace=True)
        self.all_resource_price = pd.DataFrame({'years': demand_df.index})
        self.all_resource_price.set_index('years', inplace=True)
        self.all_resource_use = pd.DataFrame({'years': demand_df.index})
        self.all_resource_use.set_index('years', inplace=True)
        self.all_resource_production = pd.DataFrame({'years': demand_df.index})
        self.all_resource_production.set_index('years', inplace=True)
        self.all_resource_ratio_prod_demand= pd.DataFrame({'years': demand_df.index})
        self.all_resource_ratio_prod_demand.set_index('years', inplace=True)
        self.all_resource_ratio_usable_demand= pd.DataFrame({'years': demand_df.index})
        self.all_resource_ratio_usable_demand.set_index('years', inplace=True)

        for resource in list(AllResourceModel.RESOURCE_LIST):
            self.all_resource_price[resource] = np.linspace(0, 0, len(self.all_resource_price.index))
            self.all_resource_production[resource] = np.linspace(0, 0, len(self.all_resource_production.index))
            self.all_resource_stock[resource] = np.linspace(0, 0, len(self.all_resource_stock.index))
            self.all_resource_use[resource] = np.linspace(0, 0, len(self.all_resource_use.index))
            self.all_resource_ratio_prod_demand[resource] = np.linspace(0, 0, len(self.all_resource_ratio_prod_demand.index))
            self.all_resource_ratio_usable_demand[resource] = np.linspace(0, 0, len(self.all_resource_ratio_usable_demand.index))
            data_frame_price= deepcopy(input_dict[f'{resource}.resource_price'])

            #set index:
            if 'years' in data_frame_price.columns:
                data_frame_price.set_index('years', inplace=True)
            data_frame_production = deepcopy(input_dict[f'{resource}.predictible_production'])
            if 'years' in data_frame_production.columns:
                data_frame_production.set_index('years', inplace=True)
            data_frame_stock = deepcopy(input_dict[f'{resource}.resource_stock'])
            if 'years' in data_frame_stock.columns:
                data_frame_stock.set_index('years', inplace=True)
            data_frame_use = deepcopy(input_dict[f'{resource}.use_stock'])
            if 'years' in data_frame_use.columns:
                data_frame_use.set_index('years', inplace=True)
            self.resource_demand = deepcopy(input_dict[f'All_Demand'])
            if 'years' in self.resource_demand.columns:
                self.resource_demand.set_index('years', inplace=True)
            self.all_resource_price[resource] = data_frame_price['price']

            #add the different resource in one dataframe for production, stock and use resource per year:
            for types in data_frame_use:
                if types != 'years':
                    self.all_resource_production[resource] = self.all_resource_production[resource] + data_frame_production[types]
                    self.all_resource_stock[resource]=self.all_resource_stock[resource]+data_frame_stock[types]
                    self.all_resource_use[resource] = self.all_resource_use[resource] + data_frame_use[types]

            #conversion in Mt of the different resource:
            if resource=='natural_gas_resource':
                self.all_resource_production[resource] = self.all_resource_production[resource]*self.gas_conversion
                self.all_resource_stock[resource]=self.all_resource_stock[resource]*self.gas_conversion
                self.all_resource_use[resource] = self.all_resource_use[resource]*self.gas_conversion
                self.all_resource_price[resource]=self.all_resource_price[resource]*self.price_gas_conversion
            if resource=='uranium_resource':
                self.all_resource_production[resource] = self.all_resource_production[resource]*self.uranium_conversion
                self.all_resource_stock[resource]=self.all_resource_stock[resource]*self.uranium_conversion
                self.all_resource_use[resource] = self.all_resource_use[resource]*self.uranium_conversion
                self.all_resource_price[resource]=self.all_resource_price[resource]*self.price_uranium_conversion
            if resource=='coal_resource':
                self.all_resource_price[resource]=self.all_resource_price[resource]*self.price_coal_conversion
            if resource=='oil_resource':
                self.all_resource_price[resource]=self.all_resource_price[resource]*self.price_oil_conversion

            #compute ratio production and usable resource VS demand:
            self.all_resource_ratio_usable_demand[resource]= self.all_resource_use[resource].div(self.resource_demand[resource])

        #price assignment for non modeled resource:
        data_frame_other_resource_price=deepcopy(input_dict['non_modeled_resource_price'])
        data_frame_other_resource_price.set_index('years', inplace=True)
        for resource_non_modeled in data_frame_other_resource_price :
            if resource_non_modeled not in AllResourceModel.RESOURCE_LIST:
                self.all_resource_price[resource_non_modeled] = data_frame_other_resource_price[resource_non_modeled]

    def get_derivative_all_resource(self,inputs_dict, resource_type):
        """ Compute derivative of total stock regarding year demand
        """
        grad_stock=pd.DataFrame()
        grad_use=pd.DataFrame()
        grad_price=pd.DataFrame()
        if resource_type=='natural_gas_resource':
            grad_stock=self.gas_conversion*np.identity(len(inputs_dict[f'{resource_type}.resource_stock'].index))
            grad_use= self.gas_conversion*np.identity(len(inputs_dict[f'{resource_type}.use_stock'].index))
        elif resource_type=='uranium_resource':
            grad_stock=self.uranium_conversion*np.identity(len(inputs_dict[f'{resource_type}.resource_stock'].index))
            grad_use= self.uranium_conversion*np.identity(len(inputs_dict[f'{resource_type}.use_stock'].index))
        else:
            grad_stock=np.identity(len(inputs_dict[f'{resource_type}.resource_stock'].index))
            grad_use= np.identity(len(inputs_dict[f'{resource_type}.use_stock'].index))
        if resource_type=='natural_gas_resource':
            grad_price= self.price_gas_conversion*np.identity(len(inputs_dict[f'{resource_type}.resource_price'].index))
        elif resource_type=='oil_resource':
            grad_price=self.price_oil_conversion*np.identity(len(inputs_dict[f'{resource_type}.resource_price'].index))
        elif resource_type=='uranium_resource':
            grad_price= self.price_uranium_conversion*np.identity(len(inputs_dict[f'{resource_type}.resource_price'].index))
        elif resource_type=='coal_resource':
            grad_price= self.price_coal_conversion*np.identity(len(inputs_dict[f'{resource_type}.resource_price'].index))
        else:
            grad_price= np.identity(len(inputs_dict[f'{resource_type}.resource_price'].index))

        return grad_price,grad_use,grad_stock

    def get_derivative_ratio(self,inputs_dict, resource_type, grad_use,output_dict):
        
        resource_use=output_dict[AllResourceModel.All_RESOURCE_USE][resource_type]
        demand=inputs_dict['All_Demand'][resource_type]
        identity_neg=np.diag(np.linspace(-1, -1, len(inputs_dict['All_Demand'].index)))
        grad_use_ratio_on_demand=(resource_use/(demand**2).values).values*identity_neg
        grad_use_ratio_on_use=grad_use/(demand.values)

        return grad_use_ratio_on_use, grad_use_ratio_on_demand
