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

from energy_models.core.stream_type.resources_models.methanol import Methanol
from energy_models.core.stream_type.resources_models.natural_oil import NaturalOil
from energy_models.core.stream_type.resources_models.oil import CrudeOil
from energy_models.core.stream_type.resources_models.oxygen import Oxygen
from energy_models.core.stream_type.resources_models.potassium_hydroxide import PotassiumHydroxide
from energy_models.core.stream_type.resources_models.sodium_hydroxide import SodiumHydroxide
from climateeconomics.sos_wrapping.sos_wrapping_resources.sos_wrapping_uranium_resource.uranium_resource_model.uranium_resource_disc import UraniumDiscipline
from climateeconomics.sos_wrapping.sos_wrapping_resources.sos_wrapping_coal_resource.coal_resource_model.coal_resource_disc import CoalDiscipline
from climateeconomics.sos_wrapping.sos_wrapping_resources.sos_wrapping_gas_resource.gas_resource_model.gas_resource_disc import GasDiscipline
from climateeconomics.sos_wrapping.sos_wrapping_resources.sos_wrapping_oil_resource.oil_resource_model.oil_resource_disc import OilDiscipline
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


class AllResourceModel():
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
    RESOURCE_LIST = [OilDiscipline.resource_name, UraniumDiscipline.resource_name,
                     GasDiscipline.resource_name, CoalDiscipline.resource_name]
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
        self.resource_demand = None
        self.set_data()
        self.price_gas_conversion = 1.379 * 35310700 * 10**-6
        self.price_uranium_conversion = (1 / 0.001102) * 0.907185
        self.price_oil_conversion = 7.33
        self.price_coal_conversion = 0.907185
        self.gas_conversion = 1 / 1.379
        self.uranium_conversion = 10**-6

    def set_data(self):
        self.year_start = self.param[AllResourceModel.YEAR_START]
        self.year_end = self.param[AllResourceModel.YEAR_END]

    def compute(self, input_dict):

        demand_df = deepcopy(input_dict['All_Demand'])
        demand_df.index = demand_df['years'].values
        # initialisation of dataframe:
        self.all_resource_stock = pd.DataFrame({'years': demand_df.index})
        self.all_resource_stock.set_index('years', inplace=True)
        self.all_resource_price = pd.DataFrame({'years': demand_df.index})
        self.all_resource_price.set_index('years', inplace=True)
        self.all_resource_use = pd.DataFrame({'years': demand_df.index})
        self.all_resource_use.set_index('years', inplace=True)
        self.all_resource_production = pd.DataFrame({'years': demand_df.index})
        self.all_resource_production.set_index('years', inplace=True)
        self.all_resource_ratio_prod_demand = pd.DataFrame(
            {'years': demand_df.index})
        self.all_resource_ratio_prod_demand.set_index('years', inplace=True)
        self.all_resource_ratio_usable_demand = pd.DataFrame(
            {'years': demand_df.index})
        self.all_resource_ratio_usable_demand.set_index('years', inplace=True)

        for resource in list(AllResourceModel.RESOURCE_LIST):
            self.all_resource_price[resource] = np.linspace(
                0, 0, len(self.all_resource_price.index))
            self.all_resource_production[resource] = np.linspace(
                0, 0, len(self.all_resource_production.index))
            self.all_resource_stock[resource] = np.linspace(
                0, 0, len(self.all_resource_stock.index))
            self.all_resource_use[resource] = np.linspace(
                0, 0, len(self.all_resource_use.index))
            self.all_resource_ratio_prod_demand[resource] = np.linspace(
                0, 0, len(self.all_resource_ratio_prod_demand.index))
            self.all_resource_ratio_usable_demand[resource] = np.linspace(
                0, 0, len(self.all_resource_ratio_usable_demand.index))
            data_frame_price = deepcopy(
                input_dict[f'{resource}.resource_price'])

            # set index:
            if 'years' in data_frame_price.columns:
                data_frame_price.set_index('years', inplace=True)
            data_frame_production = deepcopy(
                input_dict[f'{resource}.predictible_production'])
            if 'years' in data_frame_production.columns:
                data_frame_production.set_index('years', inplace=True)
            data_frame_stock = deepcopy(
                input_dict[f'{resource}.resource_stock'])
            if 'years' in data_frame_stock.columns:
                data_frame_stock.set_index('years', inplace=True)
            data_frame_use = deepcopy(input_dict[f'{resource}.use_stock'])
            if 'years' in data_frame_use.columns:
                data_frame_use.set_index('years', inplace=True)
            self.resource_demand = deepcopy(input_dict[f'All_Demand'])
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
            if resource == GasDiscipline.resource_name:
                self.all_resource_production[resource] = self.all_resource_production[resource] * \
                    self.gas_conversion
                self.all_resource_stock[resource] = self.all_resource_stock[resource] * \
                    self.gas_conversion
                self.all_resource_use[resource] = self.all_resource_use[resource] * \
                    self.gas_conversion
                self.all_resource_price[resource] = self.all_resource_price[resource] * \
                    self.price_gas_conversion
            if resource == UraniumDiscipline.resource_name:
                self.all_resource_production[resource] = self.all_resource_production[resource] * \
                    self.uranium_conversion
                self.all_resource_stock[resource] = self.all_resource_stock[resource] * \
                    self.uranium_conversion
                self.all_resource_use[resource] = self.all_resource_use[resource] * \
                    self.uranium_conversion
                self.all_resource_price[resource] = self.all_resource_price[resource] * \
                    self.price_uranium_conversion
            if resource == CoalDiscipline.resource_name:
                self.all_resource_price[resource] = self.all_resource_price[resource] * \
                    self.price_coal_conversion
            if resource == OilDiscipline.resource_name:
                self.all_resource_price[resource] = self.all_resource_price[resource] * \
                    self.price_oil_conversion

            # compute ratio production and usable resource VS demand:
            demand_limited = compute_func_with_exp_min(
                np.array(self.resource_demand[resource].values), 1.0e-10)
            self.all_resource_ratio_usable_demand[resource] = self.all_resource_use[resource].values / demand_limited

        # price assignment for non modeled resource:
        data_frame_other_resource_price = deepcopy(
            input_dict['non_modeled_resource_price'])
        data_frame_other_resource_price.set_index('years', inplace=True)
        for resource_non_modeled in data_frame_other_resource_price:
            if resource_non_modeled not in AllResourceModel.RESOURCE_LIST:
                self.all_resource_price[resource_non_modeled] = data_frame_other_resource_price[resource_non_modeled]

        self.all_resource_co2_emissions = self.get_co2_emissions(list(demand_df['years'].values),
                                                                 list(
            AllResourceModel.RESOURCE_LIST),
            list(AllResourceModel.NON_MODELED_RESOURCE_LIST))

    def get_derivative_all_resource(self, inputs_dict, resource_type):
        """ Compute derivative of total stock regarding year demand
        """
        grad_stock = pd.DataFrame()
        grad_use = pd.DataFrame()
        grad_price = pd.DataFrame()
        if resource_type == GasDiscipline.resource_name:
            grad_stock = self.gas_conversion * \
                np.identity(
                    len(inputs_dict[f'{resource_type}.resource_stock'].index))
            grad_use = self.gas_conversion * \
                np.identity(
                    len(inputs_dict[f'{resource_type}.use_stock'].index))
        elif resource_type == UraniumDiscipline.resource_name:
            grad_stock = self.uranium_conversion * \
                np.identity(
                    len(inputs_dict[f'{resource_type}.resource_stock'].index))
            grad_use = self.uranium_conversion * \
                np.identity(
                    len(inputs_dict[f'{resource_type}.use_stock'].index))
        else:
            grad_stock = np.identity(
                len(inputs_dict[f'{resource_type}.resource_stock'].index))
            grad_use = np.identity(
                len(inputs_dict[f'{resource_type}.use_stock'].index))
        if resource_type == GasDiscipline.resource_name:
            grad_price = self.price_gas_conversion * \
                np.identity(
                    len(inputs_dict[f'{resource_type}.resource_price'].index))
        elif resource_type == OilDiscipline.resource_name:
            grad_price = self.price_oil_conversion * \
                np.identity(
                    len(inputs_dict[f'{resource_type}.resource_price'].index))
        elif resource_type == UraniumDiscipline.resource_name:
            grad_price = self.price_uranium_conversion * \
                np.identity(
                    len(inputs_dict[f'{resource_type}.resource_price'].index))
        elif resource_type == CoalDiscipline.resource_name:
            grad_price = self.price_coal_conversion * \
                np.identity(
                    len(inputs_dict[f'{resource_type}.resource_price'].index))
        else:
            grad_price = np.identity(
                len(inputs_dict[f'{resource_type}.resource_price'].index))

        return grad_price, grad_use, grad_stock

    def get_derivative_ratio(self, inputs_dict, resource_type, grad_use, output_dict):

        resource_use = output_dict[AllResourceModel.All_RESOURCE_USE][resource_type]
        demand = inputs_dict['All_Demand'][resource_type]
        demand_limited = compute_func_with_exp_min(
            demand.values, 1.0e-10)
        d_demand_limited = compute_dfunc_with_exp_min(
            demand.values, 1.0e-10)
        identity_neg = np.diag(
            np.linspace(-1, -1, len(inputs_dict['All_Demand'].index)))
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

        # Loop on modeled resources, retrieve CO2 emissions and create column with result
        #(No resource with modeled CO2 emissions for now)
        for resource in resource_list:
            pass

        for resource in non_modeled_resource_list:
            if resource in self.CO2_emissions_dict.keys():
                resources_CO2_emissions[resource] = np.ones(
                    len(years)) * self.CO2_emissions_dict[resource]
            else:
                resources_CO2_emissions[resource] = np.zeros(
                    len(years))

        return resources_CO2_emissions
