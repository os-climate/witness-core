'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2023/11/03 Copyright 2023 Capgemini

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

from copy import deepcopy

import numpy as np
import pandas as pd

from climateeconomics.core.tools.Hubbert_Curve import compute_Hubbert_regression
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


class ResourceModel():
    """
    Resource pyworld3
    General implementation of a resource pyworld3, to be inherited by specific models for each type of resource
    """

    resource_name = 'resource'

    # Units conversion
    conversion_factor = 1.0
    oil_barrel_to_tonnes = 6.84
    bcm_to_Mt = 1 / 1.379
    kU_to_Mt = 10 ** -6

    def __init__(self, name):
        '''
        Constructor
        '''
        self.resource_name = name
        self.resource_stock = None
        self.resource_data = None
        self.annual_price = None
        self.resource_type = None
        self.sub_resource_list = None

        # Price and Hubert pyworld3
        self.resource_price = None
        self.Q_inf = None
        self.tho = None
        self.w = None
        self.price_data = None
        self.data_year_start = None
        self.construction_resource_list = ['copper_resource']

    def configure_parameters(self, inputs_dict):
        '''
        Configure with inputs_dict from the discipline
        '''

        self.year_start = inputs_dict[GlossaryCore.YearStart]  # year start
        self.year_end = inputs_dict[GlossaryCore.YearEnd]  # year end
        self.production_start = inputs_dict['production_start']
        self.production_years = np.arange(
            self.production_start, self.year_end + 1)
        self.stock_start = inputs_dict['stock_start']
        self.resources_demand = inputs_dict['resources_demand']
        self.resource_consumed_data = inputs_dict['resource_consumed_data']
        self.lifespan = inputs_dict['lifespan']
        self.recycled_rate = inputs_dict['recycled_rate']
        self.resource_data = inputs_dict['resource_data']
        self.resource_production_data = inputs_dict['resource_production_data']
        self.resource_price_data = inputs_dict['resource_price_data']
        self.init_dataframes()
        self.sub_resource_list = [col for col in list(
            self.resource_production_data.columns) if col != GlossaryCore.Years]

        # self.resource_consumed_dict ={}
        # for resource_type in self.sub_resource_list :
        #     self.resource_consumed_dict[f'{resource_type}_consumption'] = inputs_dict['resource_consumed_data'][f'{resource_type}_consumption'].values

    def init_dataframes(self):
        '''
        Init dataframes with years
        '''
        self.years = np.arange(self.year_start, self.year_end + 1)
        self.predictable_production = pd.DataFrame(
            {GlossaryCore.Years: np.arange(self.production_start, self.year_end + 1, 1)})
        self.recycled_production = pd.DataFrame(
            {GlossaryCore.Years: self.years})
        self.total_consumption = pd.DataFrame(
            {GlossaryCore.Years: self.years})
        self.resource_stock = pd.DataFrame(
            {GlossaryCore.Years: self.years})
        self.resource_price = pd.DataFrame(
            {GlossaryCore.Years: self.years})

        # GlossaryCore.Years : self.years if lifespan is null
        self.use_stock = pd.DataFrame(
            {GlossaryCore.Years: np.insert(self.years, 0, np.arange(self.year_start - self.lifespan, self.year_start, 1))})

        '''
        Set the index as the years
        '''
        self.predictable_production.index = self.predictable_production[GlossaryCore.Years]
        self.recycled_production.index = self.recycled_production[GlossaryCore.Years]
        self.total_consumption.index = self.total_consumption[GlossaryCore.Years]
        self.resource_stock.index = self.resource_stock[GlossaryCore.Years]
        self.resource_price.index = self.resource_price[GlossaryCore.Years]
        self.use_stock.index = self.use_stock[GlossaryCore.Years]


    def configure_parameters_update(self, inputs_dict):
        '''
        Configure with inputs_dict from the discipline
        '''

        self.resources_demand = deepcopy(inputs_dict['resources_demand'])
        self.resources_demand.index = self.resources_demand[GlossaryCore.Years]
        # re adapt the dataframe to the length of the simulation
        self.resources_demand = self.resources_demand.loc[self.resources_demand[GlossaryCore.Years]
                                                          >= self.year_start]
        self.resources_demand = self.resources_demand.loc[self.resources_demand[GlossaryCore.Years]
                                                          <= self.year_end]
        self.init_dataframes()
        self.sub_resource_list = [col for col in list(
            self.resource_production_data.columns) if col != GlossaryCore.Years]

    def compute(self):

        self.compute_predictable_production()

        self.compute_stock()

        self.compute_price()

    def compute_predictable_production(self):
        '''
        For each resource_type inside resource pyworld3, compute predictable production through Hubbert regression function
        '''
        for resource_type in self.sub_resource_list:
            self.predictable_production[resource_type] = compute_Hubbert_regression(
                self.resource_production_data, self.production_years, self.production_start, resource_type)

    def compute_stock(self):
        # pylint: disable-msg=unsubscriptable-object
        # Initialize stocks df
        for resource_type in self.sub_resource_list:
            self.resource_stock[resource_type] = np.insert(
                np.zeros(len(self.years) - 1), 0, self.stock_start)
            # Concat what was consumed in the past years and what will be
            # consumed (needed for recycled materials calculations)
            self.use_stock[resource_type] = np.insert(np.zeros(len(
                self.years) - 1), 0, self.resource_consumed_data[f'{resource_type}_consumption'])
            self.recycled_production[resource_type] = np.insert(np.zeros(len(self.years) - 1), 0,
                                                                self.resource_consumed_data[f'{resource_type}_consumption'].values[0] * self.recycled_rate)

        # Select only the right resource demand and convert the demand unit if
        # needed

        self.resource_demand = self.resources_demand[[
            GlossaryCore.Years, self.resource_name]]

        self.convert_demand(self.resource_demand)

        # Sort the resource type by ascending price
        ascending_price_resource_list = list(
            self.resource_price_data.sort_values(by=['price'])['resource_type'])
        # If needed, get the global demand from the energy demand
        self.get_global_demand(self.resource_demand)
        # Turn dataframes into dict of np array for faster computation time
        resource_demand_dict = self.resource_demand.to_dict()
        predictable_production_dict = self.predictable_production.to_dict()
        resource_stock_dict = self.resource_stock.to_dict()
        use_stock_dict = self.use_stock.to_dict()
        recycled_production_dict = self.recycled_production.to_dict()

        # compute of stock per year (stock = 0 at year 0, no longer true for
        # copper)
    
        for year_demand in self.years[1:]:
            total_stock = 0.0
            demand = resource_demand_dict[self.resource_name][year_demand]

            # we take in priority the less expensive resource
            for resource_type in ascending_price_resource_list:
                total_stock = total_stock + \
                    predictable_production_dict[resource_type][year_demand]

            # chek if the stock is not empty this year:
            if total_stock > 0:
                for resource_type in ascending_price_resource_list:
                    # compute recycled quantity of the different resources
                    self.compute_recycling(
                        year_demand, resource_type, recycled_production_dict, use_stock_dict)
                    # while demand is not satisfied we use extracted and stocked
                    # resource, if there is resource in excess we stock it
                    if demand.real > 0:
                        available_resource = resource_stock_dict[resource_type][year_demand - 1] + \
                            predictable_production_dict[resource_type][year_demand] + \
                            recycled_production_dict[resource_type][year_demand] - demand
                        if available_resource.real >= 0:
                            resource_stock_dict[resource_type][year_demand] = \
                                resource_stock_dict[resource_type][year_demand - 1] + \
                                predictable_production_dict[resource_type][year_demand] + \
                                recycled_production_dict[resource_type][year_demand] - demand
                            use_stock_dict[resource_type][year_demand] = demand
                            demand = 0

                        # if there is not enough resource we use all the
                        # resource we have and we don't answer all the demand
                        else:
                            resource_stock_dict[resource_type][year_demand] = 0
                            use_stock_dict[resource_type][year_demand] = \
                                predictable_production_dict[resource_type][year_demand] + \
                                resource_stock_dict[resource_type][year_demand - 1] + \
                                recycled_production_dict[resource_type][year_demand]
                            demand = demand - \
                                use_stock_dict[resource_type][year_demand]

                    else:
                        resource_stock_dict[resource_type][year_demand] = \
                            resource_stock_dict[resource_type][year_demand - 1] + \
                            predictable_production_dict[resource_type][year_demand] +\
                                recycled_production_dict[resource_type][year_demand]
                        use_stock_dict[resource_type][year_demand] = 0
            # if the stock is empty we just use what we produced this year
            else:
                resource_stock_dict[resource_type][year_demand] = 0
                use_stock_dict[resource_type][year_demand] = 0
        self.predictable_production = pd.DataFrame.from_dict(
            predictable_production_dict)
        self.resource_stock = pd.DataFrame.from_dict(resource_stock_dict)
        self.use_stock = pd.DataFrame.from_dict(use_stock_dict) 
        self.use_stock = self.use_stock.loc[self.use_stock[GlossaryCore.Years]>= self.year_start]
        self.use_stock= self.use_stock.loc[self.use_stock[GlossaryCore.Years]<= self.year_end]
        
        self.recycled_production = pd.DataFrame.from_dict(
            recycled_production_dict)

    def compute_price(self):
        # pylint: disable-msg=unsubscriptable-object

        # dataframe initialization
        # self.resource_price = pd.DataFrame({GlossaryCore.Years: self.years})

        # for each year we calculate the price with the proportion of each
        # resource. The resource price is stored in the price data dataframe
        self.resource_price['price'] = np.zeros(len(self.years))
        self.total_consumption['production'] = np.zeros(len(self.years))

        # we compute the total consumption of one resource
        for resource_type in self.sub_resource_list:
            self.total_consumption['production'] = self.use_stock[resource_type] + \
                self.total_consumption['production']
        # Turn dataframes into dict of np array for faster computation time
        resource_price_dict = self.resource_price.to_dict()
        resource_price_data_dict = self.resource_price_data.to_dict()
        total_consumption_dict = self.total_consumption.to_dict()
        use_stock_dict = self.use_stock.to_dict()

        # we divide each resource use by the total consumption to have the
        # proportion and we multiply by the price
        ascending_price_resource_list = list(
            self.resource_price_data.sort_values(by=['price'])['resource_type'])
        for resource_type in ascending_price_resource_list:
            mask_1 = np.array(
                [val for val in use_stock_dict[resource_type].values()]) >= 0
            mask_2 = np.array(
                [val for val in total_consumption_dict['production'].values()]) != 0
            resource_type_price_idx = list(resource_price_data_dict['resource_type'].keys())[
                list(resource_price_data_dict['resource_type'].values()).index(resource_type)]
            for year in np.array([val for val in resource_price_dict[GlossaryCore.Years].values()])[(mask_1 * mask_2)]:
                resource_price_dict['price'][year] = \
                    resource_price_dict['price'][year] + \
                    use_stock_dict[resource_type][year] / \
                    total_consumption_dict['production'][year] * \
                    resource_price_data_dict['price'][resource_type_price_idx]
        self.resource_price = pd.DataFrame.from_dict(resource_price_dict)

    def convert_demand(self, demand):
        '''
        To be overloaded in specific resource models
        '''
        pass

    def compute_recycling(self, year_demand, resource_type, recycled_production_dict, use_stock_dict):
        # infrastructures have a certain lifespan, so the recycled materials
        # obtained each are those used a lifespan ago, multiplied by a
        # recycle-rate
        recycled_production_dict[resource_type][year_demand] = \
            use_stock_dict[resource_type][year_demand - self.lifespan] * self.recycled_rate

    def get_global_demand(self, demand):
        '''
        To be overloaded in specific resource models
        '''
        pass

    def compute_derivative_recycling(self, year_demand, resource_type, grad_use, grad_stock, grad_recycling) :
        """ 
        Compute derivative of reclycling regarding demand
        """
        # recycling of the current year (year_demand - self.year_start) which depends on the used stock at current_year - lifespan
        if year_demand - self.lifespan > self.year_start :
            grad_recycling[resource_type][year_demand - self.year_start] = \
                grad_use[resource_type][year_demand - self.lifespan - self.year_start] * self.recycled_rate       
           
        return grad_recycling
        

    def get_derivative_resource(self):
        """ Compute derivative of stock, used stock and price regarding demand
        """
        # pylint: disable-msg=unsubscriptable-object
        # # ------------------------------------------------
        # # gather inputs
        year_start = self.year_start
        year_end = self.year_end
        nb_years = self.year_end - self.year_start + 1
        ascending_price_resource_list = list(
            self.resource_price_data.sort_values(by=['price'])['resource_type'])
        # Turn dataframes into dict of np array for faster computation time
        resource_demand_dict = self.resource_demand.to_dict()
        predictable_production_dict = self.predictable_production.to_dict()
        resource_stock_dict = self.resource_stock.to_dict()
        # # # resource_price_data_dict = self.resource_price_data.to_dict()
        # # # total_consumption_dict = self.total_consumption.to_dict()
        self.use_stock = self.use_stock.loc[self.use_stock[GlossaryCore.Years]>= self.year_start]
        self.use_stock= self.use_stock.loc[self.use_stock[GlossaryCore.Years]<= self.year_end]
        use_stock_dict = self.use_stock.to_dict()
        recycled_production_dict = self.recycled_production.to_dict()

        # # ------------------------------------------------
        # # init gradient dict of matrix transmitted to discipline
        # # dict of matrix, one per resource_type -> ex. for Oil: {'heavy': [...], 'medium': [...]..
        # # price matrix
        # # resource production is NOT dependent of demand since it is calculated with Hubbert regression
        grad_stock = {}
        grad_price =  np.identity(nb_years) * 0
        grad_use = {}
        # # ------------------------------------------------
        # # init useful containers for calculation
        # # no_stock_year contains the last year at which there is no stock
        # # year_stock contains years at which we stored resource without demand
        # # grad_demand is used for resource use gradient calculation
        # # grad_recycling is used for stock and use gradient calculations
        # # coef_conversion is the demand multiplier for units
        grad_demand = 0
        no_stock_year = {}
        year_stock = {}
        grad_recycling = {}
    
        for resource_type in self.sub_resource_list:
            grad_stock[resource_type] = np.identity(nb_years) * 0
            grad_use[resource_type] = np.identity(nb_years) * 0
            grad_recycling[resource_type] = np.identity(nb_years) * 0 
            no_stock_year[resource_type] = 0
            year_stock[resource_type] = []
           
        # # ------------------------------------------------
        # # gradient matrix computation
        for year_demand in range(year_start + 1, year_end + 1):
            total_stock = 0
            grad_recycling = self.compute_derivative_recycling(year_demand, resource_type, grad_use, grad_stock, grad_recycling)
            # # ------------------------------------------------
            # # dealing with units
            demand = resource_demand_dict[self.resource_name][year_demand] 

            for resource_type in ascending_price_resource_list:
                total_stock = total_stock + predictable_production_dict[resource_type][
                    year_demand]

            # check if the stock is not empty this year
            if total_stock > 0:
                for resource_type in ascending_price_resource_list:
                    if demand > 0:
                        if resource_stock_dict[resource_type][year_demand - 1] \
                            + predictable_production_dict[resource_type][
                                year_demand] \
                            + recycled_production_dict[resource_type][year_demand] \
                            - demand >= 0 and resource_stock_dict[resource_type][
                                year_demand] > 0:

                            # # -----------------------------------------------
                            # # stock of resource_type and production are sufficient to fulfill demand
                            # # so we remove demand from stock and resource type use is the demand

                            # stock at year_demand depends on demand at year if stock is not zero, if demand is not zero
                            # and if the stock was not zero after this year (by recursivity, stock is stock at year n
                            # minus stock at year n-1 which is stock at year n-2 and so on, so the stock at year n depends
                            # on all previous year unless the stock is empty at
                            # a given year).  
                            # Recycling depends on the use_stock of lifespan ago 

                           
                            for year in range(year_start + 1, year_demand + 1):
                                
                                if resource_stock_dict[resource_type][
                                    year] > 0 and \
                                        resource_demand_dict[self.resource_name][year] != 0 and \
                                        year > no_stock_year[resource_type]:
                                    
                                    
                                    grad_stock[resource_type][
                                        year_demand - year_start, year - year_start] = - self.conversion_factor 
                                if self.lifespan != 0 :
                                    if year_demand - self.lifespan > self.year_start and year <year_demand - self.lifespan:
                                        grad_stock[resource_type][year_demand - year_start, year - year_start]  = \
                                            grad_stock[resource_type][year_demand -1 - year_start, year - year_start]            

                                if year_demand == year:
                                    grad_use[resource_type][
                                        year_demand - year_start, year - year_start] = self.conversion_factor
                                # resource use depends on previous year demand if at the year considered demand is not zero
                                # and if we stored resource type without demand
                                elif resource_demand_dict[self.resource_name][year] != 0 and year in year_stock[resource_type]:
                                    grad_use[resource_type][year_demand -
                                                            year_start, year - year_start] = grad_demand 

                            grad_stock[resource_type][year_demand - year_start] += grad_recycling[resource_type][year_demand - year_start]
                            demand = 0
                            grad_demand = 0
                            year_stock[resource_type] = []
                        else:
                            # # -----------------------------------------------
                            # # stock of resource_type + production + recycling are not sufficient to fulfill demand
                            # # so we use all the stock we had at previous year, the current year production, and the current year recycled production
                            # # and remove it from the demand
                            # # then use the next resource type (by ascending order of price)
                            # # we store the no_stock_year, last year at which there is no stock

                            no_stock_year[resource_type] = year_demand
                            grad_use[resource_type][year_demand - year_start] = grad_stock[resource_type][
                                year_demand - year_start - 1] 
                            grad_use[resource_type][year_demand - year_start, year_demand - self.lifespan - year_start] += \
                                grad_recycling[resource_type][year_demand - year_start, year_demand - self.lifespan - year_start]
                            # grad_use[resource_type][year_demand - year_start] += \
                            #     grad_recycling[resource_type][year_demand - year_start]
                            demand = demand - \
                                use_stock_dict[resource_type][year_demand]
                            # if no stock at previous year grad_demand = 0
                            if resource_stock_dict[resource_type][
                                    year_demand - 1] > 0:
                                grad_demand = self.conversion_factor
                    else:
                        # # ------------------------------------------------
                        # # demand is zero or has been fulfilled by cheaper resources types
                        # # stock equal previous year stock + production + recycled production
                        # # we store all years at which we stored resource without demand
                        grad_stock[resource_type][year_demand - year_start] = grad_stock[resource_type][
                            year_demand - year_start - 1] 
                        grad_stock[resource_type][year_demand - year_start, year_demand -self.lifespan - year_start] += \
                            grad_recycling[resource_type][year_demand - year_start, year_demand - self.lifespan - year_start]
                        year_stock[resource_type].append(year_demand)
        

        grad_price = self.get_d_price_d_demand(year_start, year_end, nb_years, grad_use, grad_price)
        
        return grad_stock, grad_price, grad_use, grad_recycling


    def get_d_price_d_demand (self, year_start, year_end, nb_years, grad_use, grad_price):
        # pylint: disable-msg=unsubscriptable-object
        ascending_price_resource_list = list(
            self.resource_price_data.sort_values(by=['price'])['resource_type'])

        # Turn dataframes into dict of np array for faster computation time
        resource_price_data_dict = self.resource_price_data.to_dict()
        total_consumption_dict = self.total_consumption.to_dict()
        use_stock_dict = self.use_stock.to_dict()

        # # ------------------------------------------------
        # # init useful containers for calculation
        # # grad_total_consumption is used for price gradient calculation
        grad_total_consumption = np.identity(nb_years) * 0

        for year_demand in range(year_start + 1, year_end + 1):
            # # ------------------------------------------------
            # # total consumption -> use stock + production
            for resource_type in ascending_price_resource_list:
                grad_total_consumption[year_demand -
                                        year_start] += grad_use[resource_type][year_demand - year_start]

            # # ------------------------------------------------
            # # price is u/v function with u = use and v = total consumption
            for resource_type in ascending_price_resource_list:
                for year in range(year_start + 1, year_end + 1):
                    # # ------------------------------------------------
                    # # price is u/v function with u = use and v = total consumption
                    # # price gradient is (u'v - uv') / v^2
                    if total_consumption_dict['production'][year_demand] != 0:
                        resource_type_price_idx = list(resource_price_data_dict['resource_type'].keys())[
                            list(resource_price_data_dict['resource_type'].values()).index(resource_type)]
                        grad_price[year_demand - year_start, year - year_start] += \
                            resource_price_data_dict['price'][resource_type_price_idx]\
                            * (grad_use[resource_type][year_demand - year_start, year - year_start]
                                * total_consumption_dict['production'][year_demand]
                                - use_stock_dict[resource_type][year_demand]
                                * grad_total_consumption[year_demand - year_start, year - year_start])\
                            / (total_consumption_dict['production'][year_demand]) ** 2
        return grad_price
        