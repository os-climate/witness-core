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

class OrderOfMagnitude():
    KILO = 'k'
    #USD_PER_USton = 'USD/USton'
    # MILLION_TONNES='million_tonnes'

    magnitude_factor = {
        KILO: 10 ** 3
        # USD_PER_USton:1/0.907
        #MILLION_TONNES: 10**6
    }


class ResourceModel():
    """
    Resource model
    """

    YEAR_START = 'year_start'
    PRODUCTION_START = 'production_start'
    YEAR_END = 'year_end'
    DEMAND = 'All_Demand'
    PAST_PRODUCTION = 'past_production'
    RESOURCE_STOCK = 'resource_stock'
    RESOURCE_PRICE = 'resource_price'
    USE_STOCK = 'use_stock'
    PRODUCTION = 'predictible_production'

    def __init__(self, param):
        '''
        Constructor
        '''
        self.param = param
        self.resource_stock = None
        self.resource_data_file = 'resource_data.csv'
        self.data_dir = join(dirname(__file__), 'resources_data')
        self.resource_data = None
        self.import_resource_data()
        self.set_data()
        self.annual_price = None

        # Price and Hubert model
        self.resource_price = None
        self.Q_inf = None
        self.tho = None
        self.w = None
        self.predictible_production = pd.DataFrame(
            {'years': np.arange(self.production_start, self.year_end + 1, 1)})
        self.predictible_production.set_index('years', inplace=True)
        self.price_data = None
        self.oil_barrel_to_tonnes = 6.84
        self.total_consumption = pd.DataFrame(
            {'years': np.arange(self.year_start, self.year_end + 1, 1)})
        self.total_consumption.set_index('years', inplace=True)
        self.data_year_start = None

        # unit conversion
        self.bcm_to_Mt = 1 / 1.379
        self.kU_to_Mt = 10**-6

    def set_data(self):
        self.production_start = self.param[ResourceModel.PRODUCTION_START]
        self.year_start = self.param[ResourceModel.YEAR_START]
        self.year_end = self.param[ResourceModel.YEAR_END]

    def import_resource_data(self):
        data_file = join(self.data_dir, self.resource_data_file)
        self.resource_data = pd.read_csv(data_file)

    def compute(self, resource_demand, resource, regression_year_start):

        self.compute_extraction(resource, regression_year_start)

        # Computation methods
        self.resource_demand = deepcopy(resource_demand)
        self.resource_demand.set_index('years', inplace=True)

        # Initialize dataframe
        self.resource_stock = pd.DataFrame(
            {'years': self.resource_demand.index})
        self.resource_stock.set_index('years', inplace=True)
        self.resource_price = pd.DataFrame(
            {'years': self.resource_demand.index})
        self.resource_price.set_index('years', inplace=True)
        self.use_stock = pd.DataFrame(
            {'years': self.resource_demand.index})
        self.use_stock.set_index('years', inplace=True)

        # first year computing
        resource_data = deepcopy(self.resource_data)
        resource_data.set_index('resource_type', inplace=True)
        for resource_type in self.price_data.index:
            year_start = self.resource_stock.index.min()
            year_end = self.resource_stock.index.max()
            self.resource_stock.loc[year_start, resource_type] = 0
            self.use_stock.loc[year_start,
                               resource_type] = self.data_year_start.loc[0, resource_type]
            year_start_compute = year_start + 1
            year_end_compute = year_end + 1

         # compute of stock per year
        for year_demand in range(year_start_compute, year_end_compute):
            total_stock = 0
            demand = 0
            # unit conversion
            if resource == 'natural_gas_resource':
                demand = self.resource_demand.loc[year_demand,
                                                  resource] / self.bcm_to_Mt
            elif resource == 'uranium_resource':
                demand = self.resource_demand.loc[year_demand,
                                                  resource] / self.kU_to_Mt
            else:
                demand = self.resource_demand.loc[year_demand, resource]
            # we take in priority the less expensive resource, the resource are
            # store in the index of price data in the ascending order of price
            for resource_type in self.price_data.index:
                total_stock = total_stock + \
                    self.predictible_production.loc[year_demand, resource_type]

            # chek if the stock is not empty this year:
            if total_stock > 0:
                for resource_type in self.price_data.index:
                    # while demand is not satified we use extracted and stocked
                    # resource, if there is resource in excess we stock it
                    if demand.real > 0:
                        if self.resource_stock.loc[year_demand - 1, resource_type] + \
                                self.predictible_production.loc[year_demand, resource_type] - demand.real >= 0:
                            self.resource_stock.loc[year_demand, resource_type] = self.resource_stock.loc[
                                year_demand - 1, resource_type] + \
                                self.predictible_production.loc[
                                year_demand, resource_type] - demand
                            self.use_stock.loc[year_demand,
                                               resource_type] = demand
                            demand = 0

                        # if there is not enough resource we use all the
                        # resource we have and we don't answer all the demand
                        else:
                            self.resource_stock.loc[year_demand,
                                                    resource_type] = 0
                            self.use_stock.loc[year_demand, resource_type] = self.predictible_production.loc[
                                year_demand, resource_type] + self.resource_stock.loc[year_demand - 1, resource_type]
                            demand = demand - \
                                self.use_stock.loc[year_demand, resource_type]

                    else:
                        self.resource_stock.loc[year_demand, resource_type] = self.resource_stock.loc[
                            year_demand - 1, resource_type] + self.predictible_production.loc[year_demand, resource_type]
                        self.use_stock.loc[year_demand, resource_type] = 0
            # if the stock is empty we just use what we produced this year
            else:
                self.resource_stock.loc[year_demand, resource_type] = 0
                self.use_stock.loc[year_demand, resource_type] = 0
        # go to the price computing
        self.compute_price()

    def compute_price(self):

        # dataframe initialization
        self.resource_price = pd.DataFrame({'years': self.use_stock.index})
        self.resource_price.set_index('years', inplace=True)
        begin_at_year = self.resource_price.index.min()
        end_at_year = self.resource_price.index.max() + 1

        # for each year we calculate the price with the proportion of each
        # resource. The resource price is stored in the price data dataframe
        for year in range(begin_at_year, end_at_year):

            self.resource_price.loc[year, 'price'] = 0
            self.total_consumption.loc[year, 'production'] = 0

            # we compute the total consumption of one resource
            for resource in self.price_data.index:
                self.total_consumption.loc[year, 'production'] = self.use_stock.loc[year,
                                                                                    resource] + self.total_consumption.loc[year, 'production']
            # we divide each resource use by the total consumption to have the
            # proportion and we multiply by the price
            for resource_type in self.price_data.index:
                if self.use_stock.loc[year, resource_type] >= 0 and self.total_consumption.loc[year, 'production'] != 0:
                    self.resource_price.loc[year, 'price'] = self.resource_price.loc[year, 'price'] + self.use_stock.loc[year,
                                                                                                                         resource_type] / self.total_consumption.loc[year, 'production'] * self.price_data.loc[resource_type, 'price']

    def compute_extraction(self, resource, year_start):
        # production computing per resource type:
        for i in self.resource_data.index:
            if resource == self.resource_data.loc[i, 'resource_type']:
                # get the file of one resource past production with each types
                # of this resource production, go to hubbert curve function
                self.Hubbert_curve(
                    self.resource_data.loc[i, 'production_file_name'], year_start, resource)
                # year start consumption
                data_file = join(
                    self.data_dir, self.resource_data.loc[i, 'year_start_file'])
                self.data_year_start = pd.read_csv(data_file)
                # go to the function which extract the file of one resource
                # price
                self.import_extraction_price(
                    self.resource_data.loc[i, 'price_file_name'])

    def import_extraction_price(self, price_file_name):
        data_file = join(self.data_dir, price_file_name)
        self.price_data = pd.read_csv(data_file)
        self.price_data.set_index('resource_type', inplace=True)

    def Hubbert_curve(self, production_file_name, year_start, resource):
        data_file = join(self.data_dir, production_file_name)
        self.past_production = pd.read_csv(data_file)
        if resource != 'uranium_resource':
            for resource_type in self.past_production:
                if resource_type != 'years' :
                    year_start_curve = year_start
                    self.compute_Hubbert_regression(
                        resource_type, self.past_production, year_start)
        else:
            for resource_type in self.past_production:
                year_start_curve = 1970
                if resource_type == 'uranium_40':
                    self.compute_Hubbert_regression(
                        'uranium_40', self.past_production, year_start)

                elif resource_type != 'years':
                    self.predictible_production[resource_type] = np.linspace(
                        0, 0, len(self.predictible_production.index))

            current_year = 2020
            production_sample = self.predictible_production.loc[self.predictible_production.index
                                                                >= year_start_curve]
            for year in self.predictible_production.index:
                if year > current_year:
                    self.predictible_production.loc[year, 'uranium_80'] = production_sample.loc[year - (current_year - year_start_curve), 'uranium_40'] * (
                        1243900 - 744500) / (self.predictible_production.loc[current_year, 'uranium_40'] - self.predictible_production.loc[year_start, 'uranium_40'] + 744500)
                    self.predictible_production.loc[year, 'uranium_130'] = production_sample.loc[year - (current_year - year_start_curve), 'uranium_40'] * (
                        3791700 - 1243900) / (self.predictible_production.loc[current_year, 'uranium_40'] - self.predictible_production.loc[year_start, 'uranium_40'] + 744500)
                    self.predictible_production.loc[year, 'uranium_260'] = production_sample.loc[year - (current_year - year_start_curve), 'uranium_40'] * (
                        4723700 - 3791700) / (self.predictible_production.loc[current_year, 'uranium_40'] - self.predictible_production.loc[year_start, 'uranium_40'] + 744500)

    def compute_Hubbert_regression(self, resource_type, production_data, year_start):

        # initialization
        self.cumulative_production = pd.DataFrame(
            {'years': self.past_production['years']})
        self.ratio_P_by_Q = pd.DataFrame({'years': production_data['years']})

        # Cf documentation for the hubbert curve computing
        Q = 0  # Q is the cumulative production at one precise year

        # compute cumulative production, the production, and the ratio
        # dataframe for each year
        for production_year in self.past_production.index:
            P = self.past_production.loc[production_year, resource_type]
            self.cumulative_production.loc[production_year,
                                           resource_type] = Q + P
            Q = self.cumulative_production.loc[production_year,  resource_type]
            self.ratio_P_by_Q.loc[production_year, resource_type] = P / Q

        # keep only the part you want to make a regression on
        cumulative_sample = self.cumulative_production.loc[
            self.cumulative_production['years'] >= year_start]

        ratio_sample = self.ratio_P_by_Q.loc[self.ratio_P_by_Q['years'] >= year_start]

        fit = np.polyfit(
            cumulative_sample[resource_type], ratio_sample[resource_type], 1)

        self.w = fit[1]  # imaginary frequency

        # sum of the available and recoverable reserve (predict by Hubbert
        # model from the start of the exploitation to the end)
        self.Q_inf = -1 * (self.w / fit[0])

        self.tho = 0  # year of resource peak

        # compute of all the possible values of Tho according to Q and P and
        # take the mean values
        for ind in cumulative_sample.index:
            self.tho = self.tho + np.log((self.Q_inf / cumulative_sample.loc[ind, resource_type] - 1) * np.exp(
                cumulative_sample.loc[ind, 'years'] * self.w)) * (1 / self.w)
        self.tho = self.tho / len(cumulative_sample.index)

        # compute hubbert curve values
        self.compute_Hubbert_curve(resource_type)

    def compute_Hubbert_curve(self, resource_type):
        for year in self.predictible_production.index:
            self.predictible_production.loc[year, resource_type] = self.Q_inf * self.w * (
                (1 / (np.exp((-(self.w / 2)) * (self.tho - year)) + np.exp((self.w / 2) * (self.tho - year))))**2)

    def get_derivative_resource(self, resource):
        """ Compute derivative of stock, used stock and price regarding demand
        """
        # # ------------------------------------------------
        # # gather inputs
        year_start = self.year_start
        year_end = self.year_end
        nb_years = self.year_end - self.year_start + 1
        stock = deepcopy(self.resource_stock)
        use_stock = deepcopy(self.use_stock)
        production = deepcopy(self.predictible_production)
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
        for resource_type in self.price_data.index:
            grad_stock[resource_type] = np.identity(nb_years) * 0
            grad_use[resource_type] = np.identity(nb_years) * 0
            no_stock_year[resource_type] = 0
            year_stock[resource_type] = []
        coef_conversion = 0
        # # ------------------------------------------------
        # # gradient matrix computation
        for year_demand in range(year_start + 1, year_end + 1):
            total_stock = 0
            # # ------------------------------------------------
            # # dealing with units
            if resource == 'natural_gas_resource':
                demand = self.resource_demand.loc[year_demand,resource] / self.bcm_to_Mt
                coef_conversion = 1 / self.bcm_to_Mt
            elif resource == 'uranium_resource':
                demand = self.resource_demand.loc[year_demand,resource] / self.kU_to_Mt
                coef_conversion = 1 / self.kU_to_Mt
            else:
                demand = self.resource_demand.loc[year_demand,resource]
                coef_conversion = 1
            for resource_type in self.price_data.index:
                total_stock = total_stock + \
                    self.predictible_production.loc[year_demand, resource_type]
            #check if the stock is not empty this year
            if total_stock > 0:
                for resource_type in self.price_data.index:
                    if demand > 0:
                        if stock.loc[year_demand - 1, resource_type] + production.loc[year_demand, resource_type] \
                            - demand >= 0 and stock.loc[year_demand, resource_type] > 0:
                            # # ------------------------------------------------
                            # # stock of resource_type and production are sufficient to fulfill demand
                            # # so we remove demand from stock and resource type use is the demand
            
                            # stock at year_demand depends on demand at year if stock is not zero, if demand is not zero
                            # and if the stock was not zero after this year (by recursivity, stock is stock at year n
                            # minus stock at year n-1 which is stock at year n-2 and so on, so the stock at year n depends
                            # on all previous year unless the stock is empty at a given year) 
                            for year in range(year_start + 1, year_demand + 1):

                                if stock.loc[year, resource_type] > 0 and self.resource_demand.loc[year,resource] != 0 and year > no_stock_year[resource_type]:
                                    grad_stock[resource_type][year_demand - year_start, year - year_start] = - coef_conversion                      
    
                                if year_demand == year:
                                    grad_use[resource_type][year_demand - year_start, year - year_start] = coef_conversion
                                # resource use depends on previous year demand if at the year considered demand is not zero
                                # and if we stored resource type without demand
                                elif self.resource_demand.loc[year,resource] != 0 and year in year_stock[resource_type]:
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
                            grad_use[resource_type][year_demand - year_start] = grad_stock[resource_type][year_demand - year_start - 1]
                            demand = demand - use_stock.loc[year_demand, resource_type]
                            # if no stock at previous year grad_demand = 0
                            if stock.loc[year_demand - 1, resource_type] > 0:
                                grad_demand = coef_conversion
                    else:
                        # # ------------------------------------------------
                        # # demand is zero or has been fulfilled by cheaper resources types
                        # # stock equal previous year stock + production
                        # # we store all years at which we stored resource without demand
                        grad_stock[resource_type][year_demand - year_start] = grad_stock[resource_type][year_demand - year_start - 1]
                        year_stock[resource_type].append(year_demand)
            # # ------------------------------------------------
            # # total consumption -> use stock + production
            for resource_type in self.price_data.index:
                grad_total_consumption[year_demand - year_start] += grad_use[resource_type][year_demand - year_start]
            # # ------------------------------------------------
            # # price is u/v function with u = use and v = total consumption
            for resource_type in self.price_data.index:
                for year in range(year_start + 1, year_demand + 1):
                    # # ------------------------------------------------
                    # # price is u/v function with u = use and v = total consumption
                    # # price gradient is (u'v - uv') / v^2
                    if self.total_consumption.loc[year_demand, 'production'] != 0:
                        grad_price[year_demand - year_start, year - year_start] += self.price_data.loc[resource_type, 'price'] * \
                            (grad_use[resource_type][year_demand - year_start, year - year_start] * self.total_consumption.loc[year_demand, 'production'] \
                            - self.use_stock.loc[year_demand, resource_type] * grad_total_consumption[year_demand - year_start, year - year_start]) \
                            / (self.total_consumption.loc[year_demand, 'production']) ** 2
                                
        return grad_stock, grad_price, grad_use