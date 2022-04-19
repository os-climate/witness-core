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
from os.path import join, dirname
from copy import deepcopy
from climateeconomics.core.tools.Hubbert_Curve import compute_Hubbert_regression


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
    Resource model
    General implementation of a resource model, to be inherited by specific models for each type of resource
    """

    resource_name='resource'

    #Units conversion
    conversion_factor=1.0
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

        # Price and Hubert model
        self.resource_price = None
        self.Q_inf = None
        self.tho = None
        self.w = None
        self.price_data = None
        self.data_year_start = None

    def configure_parameters(self, inputs_dict):
        '''
        Configure with inputs_dict from the discipline
        '''

        self.year_start = inputs_dict['year_start']  # year start
        self.year_end = inputs_dict['year_end']  # year end
        self.production_start = inputs_dict['production_start']
        self.production_years = np.arange(self.production_start, self.year_end+1)
        self.resources_demand = inputs_dict['resources_demand']
        self.resource_data = inputs_dict['resource_data']
        self.resource_production_data = inputs_dict['resource_production_data']
        self.resource_price_data = inputs_dict['resource_price_data']
        self.resource_year_start_data = inputs_dict['resource_year_start_data']
        self.init_dataframes()
        self.sub_resource_list = [col for col in list(self.resource_production_data.columns) if col != 'years']

    def init_dataframes(self):
        '''
        Init dataframes with years
        '''
        self.years = np.arange(self.year_start, self.year_end + 1)
        self.predictable_production = pd.DataFrame(
            {'years': np.arange(self.production_start, self.year_end + 1, 1)})
        self.total_consumption = pd.DataFrame(
            {'years': self.years})
        self.resource_stock = pd.DataFrame(
            {'years': self.years})
        self.resource_price = pd.DataFrame(
            {'years': self.years})
        self.use_stock = pd.DataFrame(
            {'years': self.years})

    def configure_parameters_update(self, inputs_dict):
        '''
        Configure with inputs_dict from the discipline
        '''

        self.resources_demand = inputs_dict['resources_demand']
        self.init_dataframes()
        self.sub_resource_list = [col for col in list(self.resource_production_data.columns) if col != 'years']

    def compute(self):

        self.compute_predictable_production()

        self.compute_stock()

        self.compute_price()

    def compute_predictable_production(self):
        '''
        For each resource_type inside resource model, compute predictable production through Hubbert regression function
        '''
        for resource_type in self.sub_resource_list:
            self.predictable_production[resource_type] = compute_Hubbert_regression(
                self.resource_production_data, self.production_years, self.production_start, resource_type)

    def compute_stock(self):
        #Initialize stocks df
        for resource_type in self.sub_resource_list:
            self.resource_stock[resource_type] = np.zeros(len(self.years))
            self.use_stock[resource_type] = np.insert(np.zeros(len(self.years)-1),0,self.resource_year_start_data.loc[0, resource_type])
        #Select only the right resource demand and convert the demand unit if needed
        self.resource_demand = self.resources_demand[['years', self.resource_name]]
        self.convert_demand(self.resource_demand)
        #Sort the resource type by ascending price
        ascending_price_resource_list=list(self.resource_price_data.sort_values(by=['price'])['resource_type'])
        # compute of stock per year (stock = 0 at year 0)
        for year_demand in self.years[1:]:
            total_stock = 0.0
            demand = self.resource_demand.loc[self.resource_demand['years']==year_demand, self.resource_name].values[0]
            # we take in priority the less expensive resource
            for resource_type in ascending_price_resource_list:
                total_stock = total_stock + \
                              self.predictable_production.loc[self.predictable_production['years']==year_demand, resource_type].values[0]

            # chek if the stock is not empty this year:
            if total_stock > 0:
                for resource_type in ascending_price_resource_list:
                    # while demand is not satisfied we use extracted and stocked
                    # resource, if there is resource in excess we stock it
                    if demand.real > 0:
                        available_resource=self.resource_stock.loc[self.resource_stock['years']==year_demand-1, resource_type].values[0] + \
                                self.predictable_production.loc[self.predictable_production['years']==year_demand, resource_type].values[0] - demand.real
                        if available_resource >= 0:
                            self.resource_stock.loc[self.resource_stock['years']==year_demand, resource_type] =\
                                self.resource_stock.loc[self.resource_stock['years']==year_demand-1, resource_type].values[0] + \
                                self.predictable_production.loc[self.predictable_production['years']==year_demand, resource_type].values[0] - demand
                            self.use_stock.loc[self.use_stock['years']==year_demand, resource_type] = demand
                            demand = 0

                        # if there is not enough resource we use all the
                        # resource we have and we don't answer all the demand
                        else:
                            self.resource_stock.loc[self.resource_stock['years']==year_demand, resource_type] = 0
                            self.use_stock.loc[self.use_stock['years']==year_demand, resource_type] = \
                                self.predictable_production.loc[self.predictable_production['years']==year_demand, resource_type].values[0] + \
                                self.resource_stock.loc[self.resource_stock['years']==year_demand-1, resource_type].values[0]
                            demand = demand - \
                                     self.use_stock.loc[self.use_stock['years']==year_demand, resource_type].values[0]

                    else:
                        self.resource_stock.loc[self.resource_stock['years']==year_demand, resource_type] =\
                            self.resource_stock.loc[self.resource_stock['years']==year_demand-1, resource_type].values[0] + \
                            self.predictable_production.loc[self.predictable_production['years']==year_demand, resource_type].values[0]
                        self.use_stock.loc[self.use_stock['years']==year_demand, resource_type] = 0
            # if the stock is empty we just use what we produced this year
            else:
                self.resource_stock.loc[self.resource_stock['years']==year_demand, resource_type] = 0
                self.use_stock.loc[self.use_stock['years']==year_demand, resource_type] = 0

    def compute_price(self):

        # dataframe initialization
        self.resource_price = pd.DataFrame({'years': self.years})

        # for each year we calculate the price with the proportion of each
        # resource. The resource price is stored in the price data dataframe
        self.resource_price['price'] = np.zeros(len(self.years))
        self.total_consumption['production'] = np.zeros(len(self.years))

        # we compute the total consumption of one resource
        for resource_type in self.sub_resource_list:
            self.total_consumption['production'] = self.use_stock[resource_type] + \
                                                             self.total_consumption['production']
        # we divide each resource use by the total consumption to have the
        # proportion and we multiply by the price
        ascending_price_resource_list = list(self.resource_price_data.sort_values(by=['price'])['resource_type'])
        for resource_type in ascending_price_resource_list:
            mask_1=(self.use_stock[resource_type] >= 0)
            mask_2=(self.total_consumption['production'] != 0)
            for year in self.resource_price['years'].loc[(mask_1*mask_2)]:
                self.resource_price.loc[self.resource_price['years']==year, 'price'] = \
                    self.resource_price.loc[self.resource_price['years']==year, 'price'].values[0] + \
                    self.use_stock.loc[self.resource_price['years']==year, resource_type].values[0] / \
                    self.total_consumption.loc[self.resource_price['years']==year, 'production'].values[0] * \
                    self.resource_price_data.loc[self.resource_price_data['resource_type']==resource_type, 'price'].values[0]

    def convert_demand(self, demand):
        '''
        To be overloaded in specific resource models
        '''
        pass


    def get_derivative_resource(self):
        """ Compute derivative of stock, used stock and price regarding demand
        """
        # # ------------------------------------------------
        # # gather inputs
        year_start = self.year_start
        year_end = self.year_end
        nb_years = self.year_end - self.year_start + 1
        stock = deepcopy(self.resource_stock)
        use_stock = deepcopy(self.use_stock)
        production = deepcopy(self.predictable_production)
        ascending_price_resource_list = list(self.resource_price_data.sort_values(by=['price'])['resource_type'])
        # # ------------------------------------------------
        # # init gradient dict of matrix transmitted to discipline
        # # dict of matrix, one per resource_type -> ex. for Oil: {'heavy': [...], 'medium': [...]..
        # # price matrix
        # # resource production is NOT dependent of demand since it is calculated with Hubbert regression
        grad_stock = {}
        grad_price = np.identity(nb_years) * 0
        grad_use = {}
        # # ------------------------------------------------
        # # init useful containers for calculation
        # # no_stock_year contains the last year at which there is no stock
        # # year_stock contains years at which we stored resource without demand
        # # grad_demand is used for resource use gradient calculation
        # # grad_total_consumption is used for price gradient calculation
        # # coef_conversion is the demand multiplier for units
        grad_demand = 0
        no_stock_year = {}
        year_stock = {}
        grad_total_consumption = np.identity(nb_years) * 0
        for resource_type in self.sub_resource_list:
            grad_stock[resource_type] = np.identity(nb_years) * 0
            grad_use[resource_type] = np.identity(nb_years) * 0
            no_stock_year[resource_type] = 0
            year_stock[resource_type] = []
        # # ------------------------------------------------
        # # gradient matrix computation
        for year_demand in range(year_start + 1, year_end + 1):
            total_stock = 0
            # # ------------------------------------------------
            # # dealing with units
            demand = self.resource_demand.loc[self.resource_demand['years']==year_demand, self.resource_name].values[0]
            for resource_type in ascending_price_resource_list:
                total_stock = total_stock + \
                              self.predictable_production.loc[self.predictable_production['years']==year_demand, resource_type].values[0]
            # check if the stock is not empty this year
            if total_stock > 0:
                for resource_type in ascending_price_resource_list:
                    if demand > 0:
                        if stock.loc[stock['years']==year_demand-1, resource_type].values[0]\
                                + production.loc[production['years']==year_demand, resource_type].values[0]\
                                - demand >= 0 and stock.loc[stock['years']==year_demand, resource_type].values[0] > 0:
                            # # ------------------------------------------------
                            # # stock of resource_type and production are sufficient to fulfill demand
                            # # so we remove demand from stock and resource type use is the demand

                            # stock at year_demand depends on demand at year if stock is not zero, if demand is not zero
                            # and if the stock was not zero after this year (by recursivity, stock is stock at year n
                            # minus stock at year n-1 which is stock at year n-2 and so on, so the stock at year n depends
                            # on all previous year unless the stock is empty at a given year)
                            for year in range(year_start + 1, year_demand + 1):
                                if stock.loc[stock['years']==year, resource_type].values[0] > 0 and\
                                    self.resource_demand.loc[self.resource_demand['years']==year, self.resource_name].values[0] != 0 and\
                                    year > no_stock_year[resource_type]:
                                    grad_stock[resource_type][
                                        year_demand - year_start, year - year_start] = - self.conversion_factor

                                if year_demand == year:
                                    grad_use[resource_type][
                                        year_demand - year_start, year - year_start] = self.conversion_factor
                                # resource use depends on previous year demand if at the year considered demand is not zero
                                # and if we stored resource type without demand
                                elif self.resource_demand.loc[self.resource_demand['years']==year, self.resource_name].values[0] != 0 and year in year_stock[resource_type]:
                                    grad_use[resource_type][year_demand - year_start, year - year_start] = grad_demand

                            demand = 0
                            grad_demand = 0
                            year_stock[resource_type] = []
                        else:
                            # # ------------------------------------------------
                            # # stock of resource_type + production are not sufficient to fulfill demand
                            # # so we use all the stock we had at previous year and the current year production
                            # # and remove it from the demand
                            # # then use the next resource type (by ascending order of price)
                            # # we store the no_stock_year, last year at which there is no stock
                            no_stock_year[resource_type] = year_demand
                            grad_use[resource_type][year_demand - year_start] = grad_stock[resource_type][
                                year_demand - year_start - 1]
                            demand = demand - use_stock.loc[self.resource_demand['years']==year_demand, resource_type].values[0]
                            # if no stock at previous year grad_demand = 0
                            if stock.loc[stock['years']==year_demand-1, resource_type].values[0] > 0:
                                grad_demand = self.conversion_factor
                    else:
                        # # ------------------------------------------------
                        # # demand is zero or has been fulfilled by cheaper resources types
                        # # stock equal previous year stock + production
                        # # we store all years at which we stored resource without demand
                        grad_stock[resource_type][year_demand - year_start] = grad_stock[resource_type][
                            year_demand - year_start - 1]
                        year_stock[resource_type].append(year_demand)
            # # ------------------------------------------------
            # # total consumption -> use stock + production
            for resource_type in ascending_price_resource_list:
                grad_total_consumption[year_demand - year_start] += grad_use[resource_type][year_demand - year_start]
            # # ------------------------------------------------
            # # price is u/v function with u = use and v = total consumption
            for resource_type in ascending_price_resource_list:
                for year in range(year_start + 1, year_end + 1):
                    # # ------------------------------------------------
                    # # price is u/v function with u = use and v = total consumption
                    # # price gradient is (u'v - uv') / v^2
                    if self.total_consumption.loc[self.total_consumption['years']==year_demand, 'production'].values[0] != 0:
                        grad_price[year_demand - year_start, year - year_start] += \
                            self.resource_price_data.loc[
                                self.resource_price_data['resource_type'] == resource_type, 'price'].values[0]\
                            * (grad_use[resource_type][year_demand - year_start, year - year_start]\
                            * self.total_consumption.loc[self.total_consumption['years']==year_demand, 'production'].values[0]\
                            - self.use_stock.loc[self.use_stock['years']==year_demand, resource_type].values[0]\
                            * grad_total_consumption[year_demand - year_start, year - year_start])\
                            / (self.total_consumption.loc[self.total_consumption['years']==year_demand, 'production'].values[0]) ** 2

        return grad_stock, grad_price, grad_use