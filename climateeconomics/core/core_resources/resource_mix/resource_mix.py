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
from energy_models.core.stream_type.resources_models.resource_glossary import (
    ResourceGlossary,
)
from sostrades_core.tools.base_functions.exp_min import (
    compute_dfunc_with_exp_min,
    compute_func_with_exp_min,
)

from climateeconomics.core.core_resources.models.coal_resource.coal_resource_disc import (
    CoalResourceDiscipline,
)
from climateeconomics.core.core_resources.models.copper_resource.copper_resource_disc import (
    CopperResourceDiscipline,
)
from climateeconomics.core.core_resources.models.natural_gas_resource.natural_gas_resource_disc import (
    NaturalGasResourceDiscipline,
)
from climateeconomics.core.core_resources.models.oil_resource.oil_resource_disc import (
    OilResourceDiscipline,
)
from climateeconomics.core.core_resources.models.uranium_resource.uranium_resource_disc import (
    UraniumResourceDiscipline,
)
from climateeconomics.glossarycore import GlossaryCore


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
    All Resource pyworld3
    """
    YEAR_START = GlossaryCore.YearStart
    YEAR_END = GlossaryCore.YearEnd
    ALL_RESOURCE_DEMAND = 'all_resource_demand'
    ALL_RESOURCE_STOCK = 'all_resource_stock'
    All_RESOURCE_USE = 'all_resource_use'
    ALL_RESOURCE_PRODUCTION = 'all_resource_production'
    ALL_RESOURCE_RECYCLED_PRODUCTION = 'all_resource_recycled_production'
    RATIO_USABLE_DEMAND = 'all_resource_ratio_usable_demand'
    ALL_RESOURCE_RATIO_PROD_DEMAND = 'all_resource_ratio_prod_demand'
    ALL_RESOURCE_CO2_EMISSIONS = 'resources_CO2_emissions'
    NON_MODELED_RESOURCE_PRICE = 'non_modeled_resource_price'
    RESOURCE_DISC_LIST = [OilResourceDiscipline, UraniumResourceDiscipline,
                          NaturalGasResourceDiscipline, CoalResourceDiscipline,
                          CopperResourceDiscipline, ]#PlatinumResourceDiscipline]

    RESOURCE_LIST = [disc.resource_name for disc in RESOURCE_DISC_LIST]

    RESOURCE_PROD_UNIT = {
        disc.resource_name: disc.prod_unit for disc in RESOURCE_DISC_LIST}

    RESOURCE_STOCK_UNIT = {
        disc.resource_name: disc.stock_unit for disc in RESOURCE_DISC_LIST}

    RESOURCE_PRICE_UNIT = {
        disc.resource_name: disc.price_unit for disc in RESOURCE_DISC_LIST}

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
        self.all_resource_recycled_production = None
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

        self.year_start = inputs_dict[GlossaryCore.YearStart]  # year start
        self.year_end = inputs_dict[GlossaryCore.YearEnd]  # year end
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
        empty_dict = {GlossaryCore.Years: self.years}
        empty_dict.update({f'{resource}': np.zeros(len(self.years))
                           for resource in self.resource_list})
        self.all_resource_stock = pd.DataFrame(empty_dict)
        self.all_resource_price = pd.DataFrame(empty_dict)
        self.all_resource_use = pd.DataFrame(empty_dict)
        self.all_resource_production = pd.DataFrame(empty_dict)
        self.all_resource_recycled_production = pd.DataFrame(empty_dict)
        self.all_resource_ratio_prod_demand = pd.DataFrame(empty_dict)
        self.all_resource_ratio_usable_demand = pd.DataFrame(empty_dict)

    def configure_parameters_update(self, inputs_dict):
        '''
        Configure with inputs_dict from the discipline
        '''

        self.resources_demand = inputs_dict['resources_demand']
        self.resources_demand = self.resources_demand.loc[self.resources_demand[GlossaryCore.Years]
                                                          >= self.year_start]
        self.resources_demand = self.resources_demand.loc[self.resources_demand[GlossaryCore.Years]
                                                          <= self.year_end]
        self.resources_demand_woratio = inputs_dict['resources_demand_woratio']
        self.resources_demand_woratio = self.resources_demand_woratio.loc[self.resources_demand_woratio[GlossaryCore.Years]
                                                                          >= self.year_start]
        self.resources_demand_woratio = self.resources_demand_woratio.loc[self.resources_demand_woratio[GlossaryCore.Years]
                                                                          <= self.year_end]
        self.resource_list = inputs_dict['resource_list']
        self.init_dataframes()

    def prepare_dataframes(self, inputs_dict):
        '''
        Set dataframes GlossaryCore.Years column to idx and concatenate inputs df
        '''
        # initialisation of dataframe:
        self.all_resource_stock.set_index(GlossaryCore.Years, inplace=True)
        self.all_resource_price.set_index(GlossaryCore.Years, inplace=True)
        self.all_resource_use.set_index(GlossaryCore.Years, inplace=True)
        self.all_resource_production.set_index(GlossaryCore.Years, inplace=True)
        self.all_resource_recycled_production.set_index(GlossaryCore.Years, inplace=True)
        self.all_resource_ratio_prod_demand.set_index(GlossaryCore.Years, inplace=True)
        self.all_resource_ratio_usable_demand.set_index(GlossaryCore.Years, inplace=True)
        for resource in self.resource_list:
            data_frame_price = deepcopy(
                inputs_dict[f'{resource}.resource_price'])
            # set index:
            if GlossaryCore.Years in data_frame_price.columns:
                data_frame_price.set_index(GlossaryCore.Years, inplace=True)
            data_frame_production = deepcopy(
                inputs_dict[f'{resource}.predictable_production'])
            if GlossaryCore.Years in data_frame_production.columns:
                data_frame_production.set_index(GlossaryCore.Years, inplace=True)
            data_frame_stock = deepcopy(
                inputs_dict[f'{resource}.resource_stock'])
            if GlossaryCore.Years in data_frame_stock.columns:
                data_frame_stock.set_index(GlossaryCore.Years, inplace=True)
            data_frame_use = deepcopy(inputs_dict[f'{resource}.use_stock'])
            if GlossaryCore.Years in data_frame_use.columns:
                data_frame_use.set_index(GlossaryCore.Years, inplace=True)
            data_frame_recycled_production = deepcopy(inputs_dict[f'{resource}.recycled_production'])
            if GlossaryCore.Years in data_frame_recycled_production.columns:
                data_frame_recycled_production.set_index(GlossaryCore.Years, inplace=True)
            self.resource_demand = deepcopy(inputs_dict['resources_demand'])
            if GlossaryCore.Years in self.resource_demand.columns:
                self.resource_demand.set_index(GlossaryCore.Years, inplace=True)
            self.all_resource_price[resource] = data_frame_price['price']

            # add the different resource in one dataframe for production, stock, recycled production
            # and use resource per year:
            for types in data_frame_use:
                if types != GlossaryCore.Years:
                    self.all_resource_production[resource] = self.all_resource_production[resource] + \
                        data_frame_production[types]
                    self.all_resource_stock[resource] = self.all_resource_stock[resource] + \
                        data_frame_stock[types]
                    self.all_resource_use[resource] = self.all_resource_use[resource] + \
                        data_frame_use[types]
                    self.all_resource_recycled_production[resource] = self.all_resource_recycled_production[resource] + \
                        data_frame_recycled_production[types]
            # conversion in Mt of the different resource:
            self.all_resource_production[resource] = self.all_resource_production[resource] * \
                self.conversion_dict[resource]['production']
            self.all_resource_stock[resource] = self.all_resource_stock[resource] * \
                self.conversion_dict[resource]['stock']
            self.all_resource_use[resource] = self.all_resource_use[resource] * \
                self.conversion_dict[resource]['stock']
            self.all_resource_recycled_production[resource] = self.all_resource_recycled_production[resource] * \
                self.conversion_dict[resource]['stock']
            self.all_resource_price[resource] = self.all_resource_price[resource] * \
                self.conversion_dict[resource]['price']
            self.resource_demand[resource] = self.resource_demand[resource] *\
                self.conversion_dict[resource]['global_demand']

    def compute(self, inputs_dict):

        self.configure_parameters_update(inputs_dict)

        self.prepare_dataframes(inputs_dict)

        self.compute_ratio()

        # price assignment for non modeled resource:
        data_frame_other_resource_price = deepcopy(
            inputs_dict['non_modeled_resource_price'])
        data_frame_other_resource_price.set_index(GlossaryCore.Years, inplace=True)
        for resource_non_modeled in data_frame_other_resource_price:
            if resource_non_modeled not in ResourceMixModel.RESOURCE_LIST:
                self.all_resource_price[resource_non_modeled] = data_frame_other_resource_price[resource_non_modeled]

        self.all_resource_co2_emissions = self.get_co2_emissions(self.years, self.resource_list,
                                                                 list(ResourceMixModel.NON_MODELED_RESOURCE_LIST))

    def compute_ratio(self):
        '''! Computes the demand_ratio dataframe.
        The ratio is calculated using the resource_use and demand WITHOUT the ratio applied
        The value of the ratio is capped to 100.0
        '''

        for resource in self.resource_list:
            # Available resources
            available_resource = deepcopy(self.all_resource_stock[resource].values) +\
                deepcopy(self.all_resource_production[resource].values) +\
                deepcopy(self.all_resource_recycled_production[resource].values)
            available_resource_limited = compute_func_with_exp_min(
                np.array(available_resource), 1.0e-10)
            # Demand without ratio
            demand_woratio = deepcopy(
                self.resources_demand_woratio[resource].values * self.conversion_dict[resource]['global_demand'])
            demand_limited = compute_func_with_exp_min(
                np.array(demand_woratio), 1.0e-10)
            self.all_resource_ratio_usable_demand[resource] = np.minimum(
                np.maximum(available_resource_limited / demand_limited, 1E-15), 1.0) * 100.0

    def get_derivative_all_resource(self, inputs_dict, resource_type):
        """ Compute derivative of total stock regarding year demand
        """
        grad_stock = pd.DataFrame()
        grad_use = pd.DataFrame()
        grad_price = pd.DataFrame()
        grad_recycling = pd.DataFrame()
        grad_demand = pd.DataFrame()
        grad_stock = self.conversion_dict[resource_type]['stock'] * \
            np.identity(
            len(inputs_dict[f'{resource_type}.resource_stock'].index))
        grad_use = self.conversion_dict[resource_type]['stock'] * \
            np.identity(
            len(inputs_dict[f'{resource_type}.use_stock'].index))
        grad_price = self.conversion_dict[resource_type]['price'] * \
            np.identity(
            len(inputs_dict[f'{resource_type}.resource_price'].index))
        grad_recycling =  self.conversion_dict[resource_type]['stock'] * \
            np.identity(
            len(inputs_dict[f'{resource_type}.recycled_production'].index))
        grad_demand = self.conversion_dict[resource_type]['global_demand'] * \
            np.identity(
            len(inputs_dict['resources_demand'].index))
        return grad_price, grad_use, grad_stock, grad_recycling, grad_demand

    def get_derivative_ratio(self, inputs_dict, resource_type, output_dict):

        resource_stock = output_dict[ResourceMixModel.ALL_RESOURCE_STOCK][resource_type]
        resource_production = output_dict[ResourceMixModel.ALL_RESOURCE_PRODUCTION][resource_type]
        resource_recycled_production = output_dict[ResourceMixModel.ALL_RESOURCE_RECYCLED_PRODUCTION][resource_type]
        # Use with ratio
        available_resource = resource_stock + resource_production + resource_recycled_production
        available_resource_limited = compute_func_with_exp_min(
            np.array(available_resource), 1.0e-10)
        d_available_resource_limited = compute_dfunc_with_exp_min(
            np.array(available_resource), 1.0e-10)
        demand = inputs_dict['resources_demand_woratio'][resource_type] * self.conversion_dict[resource_type]['global_demand']
        demand_limited = compute_func_with_exp_min(
            demand.values, 1.0e-10)
        d_demand_limited = compute_dfunc_with_exp_min(
            demand.values, 1.0e-10) 

        # If prod < cons, set the identity element for the given year to
        # the corresponding value
        d_ratio_d_stock = np.identity(len(inputs_dict['resources_demand_woratio'].index)) * 100.0 * \
            np.where((available_resource_limited <= demand_limited) * (available_resource_limited / demand_limited > 1E-15),
                     d_available_resource_limited / demand_limited,
                     0.0)
        
        d_ratio_d_recycling = np.identity(len(inputs_dict['resources_demand_woratio'].index)) * 100.0 * \
            np.where((available_resource_limited <= demand_limited) * (available_resource_limited / demand_limited > 1E-15),
                     d_available_resource_limited / demand_limited,
                     0.0)

        d_ratio_d_demand = np.identity(len(inputs_dict['resources_demand_woratio'].index)) * 100.0 * \
            np.where((available_resource_limited <= demand_limited) * (available_resource_limited / demand_limited > 1E-15),
                     -available_resource_limited * d_demand_limited * self.conversion_dict[resource_type]['global_demand'] / 
                     demand_limited ** 2 ,
                     0.0)

        return d_ratio_d_stock, d_ratio_d_demand, d_ratio_d_recycling

    def get_co2_emissions(self, years, resource_list, non_modeled_resource_list):
        '''Function to create a dataframe with the CO2 emissions for all the ressources
        For now it just create a df with set values but it should be upgraded to retrieve the CO2 emissions
        from each modeled resources
        '''
        # Create a dataframe
        resources_CO2_emissions = pd.DataFrame({GlossaryCore.Years: years})

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
