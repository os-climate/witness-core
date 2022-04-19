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

from climateeconomics.core.core_resources.models.uranium_resource.uranium_resource_disc import UraniumResourceDiscipline
from climateeconomics.core.core_resources.models.coal_resource.coal_resource_disc import CoalResourceDiscipline
from climateeconomics.core.core_resources.models.natural_gas_resource.natural_gas_resource_disc import NaturalGasResourceDiscipline
from climateeconomics.core.core_resources.models.oil_resource.oil_resource_disc import OilResourceDiscipline
from energy_models.core.stream_type.resources_models.resource_glossary import ResourceGlossary
from sos_trades_core.tools.base_functions.exp_min import compute_dfunc_with_exp_min,\
    compute_func_with_exp_min


class OrderOfMagnitude():
    KILO = 'k'
    #USD_PER_USton = 'USD/USton'
    # MILLION_TONNES='million_tonnes'

    magnitude_factor = {
        KILO: 10 ** 3
        # USD_PER_USton:1/0.907
        #MILLION_TONNES: 10**6
    }


class ResourceMixModel():
    """
    All Resource model
    """
    YEAR_START = 'year_start'
    YEAR_END = 'year_end'
    ALL_RESOURCE_DEMAND = 'all_resource_demand'
    ALL_RESOURCE_STOCK = 'all_resource_stock'
    ALL_RESOURCE_PRICE = 'resources_price'
    All_RESOURCE_USE = 'all_resource_use'
    ALL_RESOURCE_PRODUCTION = 'all_resource_production'
    RATIO_USABLE_DEMAND = 'all_resource_ratio_usable_demand'
    ALL_RESOURCE_RATIO_PROD_DEMAND = 'all_resource_ratio_prod_demand'
    ALL_RESOURCE_CO2_EMISSIONS = 'resources_CO2_emissions'
    NON_MODELED_RESOURCE_PRICE = 'non_modeled_resource_price'
    RESOURCE_LIST = [OilResourceDiscipline.resource_name, UraniumResourceDiscipline.resource_name,
                     NaturalGasResourceDiscipline.resource_name, CoalResourceDiscipline.resource_name]
    NON_MODELED_RESOURCE_LIST = list(
        set([resource['name'] for resource in ResourceGlossary.GlossaryDict.values()]).symmetric_difference(set(RESOURCE_LIST)))

    CO2_emissions_dict = {}
    for resource in ResourceGlossary.GlossaryDict.values():
        CO2_emissions_dict[resource['name']] = resource['CO2_emissions']

    def __init__(self, param):
        '''
        Constructor
        '''
        self.param = param
        self.all_resource_stock = None
        self.all_resource_price = None
        self.all_resource_use = None
        self.all_resource_production = None
        self.all_resource_ratio_prod_demand = None
        self.all_resource_ratio_usable_demand = None
        self.all_resource_co2_emissions = None
        self.resource_list = None
        self.resource_demand = None
        self.conversion_dict = None

    def configure_parameters(self, inputs_dict):
        '''
        Configure with inputs_dict from the discipline
        '''

        self.year_start = inputs_dict['year_start']  # year start
        self.year_end = inputs_dict['year_end']  # year end
        self.resource_list = inputs_dict['resource_list']
        self.non_modeled_resource_price = inputs_dict['non_modeled_resource_price']
        self.resources_demand = inputs_dict['resources_demand']
        self.resources_demand_woratio = inputs_dict['resources_demand_woratio']
        self.conversion_dict = inputs_dict['conversion_dict']
        self.init_dataframes()

    def init_dataframes(self):
        '''
        Init dataframes with years
        '''
        self.years = np.arange(self.year_start, self.year_end + 1)
        empty_dict={'years': self.years}
        empty_dict.update({f'{resource}': np.zeros(len(self.years)) for resource in self.resource_list})
        self.all_resource_stock = pd.DataFrame(empty_dict)
        self.all_resource_price = pd.DataFrame(empty_dict)
        self.all_resource_use = pd.DataFrame(empty_dict)
        self.all_resource_production = pd.DataFrame(empty_dict)
        self.all_resource_ratio_prod_demand = pd.DataFrame(empty_dict)
        self.all_resource_ratio_usable_demand = pd.DataFrame(empty_dict)

    def configure_parameters_update(self, inputs_dict):
        '''
        Configure with inputs_dict from the discipline
        '''

        self.resources_demand = inputs_dict['resources_demand']
        self.resources_demand_woratio = inputs_dict['resources_demand_woratio']
        self.resource_list = inputs_dict['resource_list']
        self.init_dataframes()

    def prepare_dataframes(self, inputs_dict):
        '''
        Set dataframes 'years' column to idx and concatenate inputs df
        '''
        # initialisation of dataframe:
        self.all_resource_stock.set_index('years', inplace=True)
        self.all_resource_price.set_index('years', inplace=True)
        self.all_resource_use.set_index('years', inplace=True)
        self.all_resource_production.set_index('years', inplace=True)
        self.all_resource_ratio_prod_demand.set_index('years', inplace=True)
        self.all_resource_ratio_usable_demand.set_index('years', inplace=True)
        for resource in self.resource_list:
            data_frame_price = deepcopy(
                inputs_dict[f'{resource}.resource_price'])
            # set index:
            if 'years' in data_frame_price.columns:
                data_frame_price.set_index('years', inplace=True)
            data_frame_production = deepcopy(
                inputs_dict[f'{resource}.predictable_production'])
            if 'years' in data_frame_production.columns:
                data_frame_production.set_index('years', inplace=True)
            data_frame_stock = deepcopy(
                inputs_dict[f'{resource}.resource_stock'])
            if 'years' in data_frame_stock.columns:
                data_frame_stock.set_index('years', inplace=True)
            data_frame_use = deepcopy(inputs_dict[f'{resource}.use_stock'])
            if 'years' in data_frame_use.columns:
                data_frame_use.set_index('years', inplace=True)
            self.resource_demand = deepcopy(inputs_dict[f'resources_demand'])
            if 'years' in self.resource_demand.columns:
                self.resource_demand.set_index('years', inplace=True)
            self.all_resource_price[resource] = data_frame_price['price']

            # add the different resource in one dataframe for production, stock
            # and use resource per year:
            for types in data_frame_use:
                if types != 'years':
                    self.all_resource_production[resource] = self.all_resource_production[resource] + \
                        data_frame_production[types]
                    self.all_resource_stock[resource] = self.all_resource_stock[resource] + \
                        data_frame_stock[types]
                    self.all_resource_use[resource] = self.all_resource_use[resource] + \
                        data_frame_use[types]
            # conversion in Mt of the different resource:
            self.all_resource_production[resource] = self.all_resource_production[resource] * \
                                                     self.conversion_dict[resource]['production']
            self.all_resource_stock[resource] = self.all_resource_stock[resource] * \
                                                self.conversion_dict[resource]['stock']
            self.all_resource_use[resource] = self.all_resource_use[resource] * \
                                              self.conversion_dict[resource]['stock']
            self.all_resource_price[resource] = self.all_resource_price[resource] * \
                                                self.conversion_dict[resource]['price']

    def compute(self, inputs_dict):

        self.configure_parameters_update(inputs_dict)

        self.prepare_dataframes(inputs_dict)

        self.compute_ratio()

        # price assignment for non modeled resource:
        data_frame_other_resource_price = deepcopy(
            inputs_dict['non_modeled_resource_price'])
        data_frame_other_resource_price.set_index('years', inplace=True)
        for resource_non_modeled in data_frame_other_resource_price:
            if resource_non_modeled not in ResourceMixModel.RESOURCE_LIST:
                self.all_resource_price[resource_non_modeled] = data_frame_other_resource_price[resource_non_modeled]

        self.all_resource_co2_emissions = self.get_co2_emissions(self.years, self.resource_list,
                                                                 list(ResourceMixModel.NON_MODELED_RESOURCE_LIST))

    def compute_ratio(self):
        """ Compute ratios
        """
        for resource in self.resource_list:
            # compute ratio production and usable resource VS demand:
            demand_limited = compute_func_with_exp_min(
                np.array(self.resources_demand_woratio[resource].values), 1.0e-10)
            self.all_resource_ratio_usable_demand[resource] = self.all_resource_use[resource].values / demand_limited

    def get_derivative_all_resource(self, inputs_dict, resource_type):
        """ Compute derivative of total stock regarding year demand
        """
        grad_stock = pd.DataFrame()
        grad_use = pd.DataFrame()
        grad_price = pd.DataFrame()
        grad_stock = self.conversion_dict[resource_type]['stock'] * \
                     np.identity(
                         len(inputs_dict[f'{resource_type}.resource_stock'].index))
        grad_use = self.conversion_dict[resource_type]['stock'] * \
                     np.identity(
                         len(inputs_dict[f'{resource_type}.use_stock'].index))
        grad_price = self.conversion_dict[resource_type]['price'] * \
                     np.identity(
                         len(inputs_dict[f'{resource_type}.resource_price'].index))
        return grad_price, grad_use, grad_stock

    def get_derivative_ratio(self, inputs_dict, resource_type, grad_use, output_dict):

        resource_use = output_dict[ResourceMixModel.All_RESOURCE_USE][resource_type]
        demand = inputs_dict['resources_demand_woratio'][resource_type]
        demand_limited = compute_func_with_exp_min(
            demand.values, 1.0e-10)
        d_demand_limited = compute_dfunc_with_exp_min(
            demand.values, 1.0e-10)
        identity_neg = np.diag(
            np.linspace(-1, -1, len(inputs_dict['resources_demand_woratio'].index)))
        # pylint: disable=unsubscriptable-object
        grad_use_ratio_on_demand = (
            resource_use * d_demand_limited.T[0] / (demand_limited**2)).values * identity_neg
        grad_use_ratio_on_use = grad_use / (demand_limited)

        return grad_use_ratio_on_use, grad_use_ratio_on_demand

    def get_co2_emissions(self, years, resource_list, non_modeled_resource_list):
        '''Function to create a dataframe with the CO2 emissions for all the ressources
        For now it just create a df with set values but it should be upgraded to retrieve the CO2 emissions
        from each modeled resources
        '''
        # Create a dataframe
        resources_CO2_emissions = pd.DataFrame({'years': years})


        for resource in non_modeled_resource_list + resource_list:
            if resource in self.CO2_emissions_dict.keys():
                resources_CO2_emissions[resource] = np.ones(
                    len(years)) * self.CO2_emissions_dict[resource]
            else:
                resources_CO2_emissions[resource] = np.zeros(
                    len(years))

        # Loop on modeled resources, retrieve CO2 emissions and create column with result
        # (No resource with modeled CO2 emissions for now)
        for resource in resource_list:
            pass

        return resources_CO2_emissions
