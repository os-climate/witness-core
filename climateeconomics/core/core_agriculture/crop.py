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
import os
from copy import deepcopy
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry

from sos_trades_core.tools.base_functions.exp_min import compute_dfunc_with_exp_min, compute_func_with_exp_min

class OrderOfMagnitude():

    KILO = 'k'
    MEGA = 'M'
    GIGA = 'G'
    TERA = 'T'

    magnitude_factor = {
        KILO: 10 ** 3,
        MEGA: 10 ** 6,
        GIGA: 10 ** 9,
        TERA: 10 ** 12
    }


class Crop():
    """
    Crop model class 
    """

    KM_2_unit = 'km2'
    HECTARE = 'ha'

    YEAR_START = 'year_start'
    YEAR_END = 'year_end'
    TIME_STEP = 'time_step'
    POPULATION_DF = 'population_df'
    DIET_DF = 'diet_df'
    KG_TO_KCAL_DICT = 'kg_to_kcal_dict'
    KG_TO_M2_DICT = 'kg_to_m2_dict'
    RED_TO_WHITE_MEAT = 'red_to_white_meat'
    MEAT_TO_VEGETABLES = 'meat_to_vegetables'
    OTHER_USE_CROP = 'other_use_crop'
    FOOD_LAND_SURFACE_DF = 'food_land_surface_df'
    INVEST_LEVEL = 'invest_level'

    min_value_invest = 1.e-12

    def __init__(self, param):
        '''
        Constructor
        '''
        self.param = param
        self.world_surface_data = None
        self.hatom2 = 0.0001
        self.m2toha = 10000.
        self.surface_df = None
        self.scaling_factor_invest_level = None

        self.set_data()
        self.create_dataframe()

    def set_data(self):
        self.year_start = self.param[Crop.YEAR_START]
        self.year_end = self.param[Crop.YEAR_END]
        self.time_step = self.param[Crop.TIME_STEP]
        years = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years = years
        self.diet_df = self.param[Crop.DIET_DF]
        self.kg_to_kcal_dict = self.param[Crop.KG_TO_KCAL_DICT]
        self.kg_to_m2_dict = self.param[Crop.KG_TO_M2_DICT]
        self.red_to_white_meat = self.param[Crop.RED_TO_WHITE_MEAT]
        self.meat_to_vegetables = self.param[Crop.MEAT_TO_VEGETABLES]
        self.other_use_crop = self.param[Crop.OTHER_USE_CROP]
        self.param_a = self.param['param_a']
        self.param_b = self.param['param_b']
        self.invest_level = self.param['invest_level']
        self.transport_cost = self.param['transport_cost']
        self.transport_margin = self.param['transport_margin']
        self.data_fuel_dict = self.param['data_fuel_dict']
        self.techno_infos_dict = self.param['techno_infos_dict']
        self.scaling_factor_invest_level = self.param['scaling_factor_invest_level']
        self.scaling_factor_techno_consumption = self.param['scaling_factor_techno_consumption']
        self.scaling_factor_techno_production = self.param['scaling_factor_techno_production']
        self.initial_age_distrib = self.param['initial_age_distrib']
        self.initial_production = self.param['initial_production']
        self.nb_years_amort_capex = 10
        self.construction_delay = 3 # default value
        self.margin = self.param['margin']
        if 'construction_delay' in self.techno_infos_dict:
            self.construction_delay = self.techno_infos_dict['construction_delay']
        else:
            print(
                f'The construction_delay data is not set for Crop : default = 3 years  ')

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''

        self.food_land_surface_df = pd.DataFrame({'years': self.years})
        self.total_food_land_surface = pd.DataFrame({'years': self.years})
        self.mix_detailed_production = pd.DataFrame({'years': self.years})
        self.mix_detailed_prices = pd.DataFrame({'years': self.years})
        self.production = pd.DataFrame({'years': self.years})
        self.land_use_required = pd.DataFrame({'years': self.years})
        self.cost_details = pd.DataFrame({'years': self.years})
        self.techno_prices = pd.DataFrame({'years': self.years})
        self.column_dict = {'red meat (Gha)': 'red meat', 'white meat (Gha)': 'white meat',
                            'milk (Gha)': 'milk', 'eggs (Gha)': 'eggs', 'rice and maize (Gha)': 'rice and maize',
                            'potatoes (Gha)': 'potatoes', 'fruits and vegetables (Gha)': 'fruits and vegetables',
                            'other (Gha)': 'other', 'total surface (Gha)': 'total surface'}
        self.techno_consumption = pd.DataFrame({'years': self.years})
        self.techno_consumption_woratio = pd.DataFrame({'years': self.years})
        self.CO2_emissions = pd.DataFrame({'years': self.years})

    def configure_parameters(self, inputs_dict):
        '''
        Configure with inputs_dict from the discipline
        '''
        if inputs_dict['margin'] is not None:
            self.margin = inputs_dict['margin'].loc[inputs_dict['margin']['years']
                                                    <= self.year_end]
        # invest level from G$ to M$
        self.scaling_factor_invest_level = inputs_dict['scaling_factor_invest_level']
        self.invest_level = inputs_dict['invest_level']
        self.invest_level['invest'] = self.invest_level['invest'] * \
            self.scaling_factor_invest_level
        if self.initial_age_distrib['distrib'].sum() > 100.001 or self.initial_age_distrib[
                'distrib'].sum() < 99.999:
            sum_distrib = self.initial_age_distrib['distrib'].sum()
            raise Exception(
                f'The distribution sum is not equal to 100 % : {sum_distrib}')

    def compute(self, population_df, temperature_df):
        ''' 
        Computation methods
        Compute the different output : updated diet, surface used (Gha), surface used (%)

        @param population_df: population from input
        @type population_df: dataframe
        @unit population_df: million of people

        @param food_land_surface_df:  Gives the surface used by each food type + the total surface used
        @type food_land_surface_df: dataframe
        @unit food_land_surface_df: Gha

        @param land_surface_percentage_df:  Gives the share of surface used by each food type
        @type land_surface_percentage_df: dataframe
        @unit land_surface_percentage_df: %

        @param updated_diet_df:  Gives the updatred diet, ie, the quantity of food for each year and each food type
        @type updated_diet_df: dataframe
        @unit updated_diet_df: kg/person/year

        '''     
        
        # construct the diet over time
        update_diet_df = self.diet_change(self.diet_df, self.red_to_white_meat,
                                          self.meat_to_vegetables)
        self.updated_diet_df = deepcopy(update_diet_df)

        # compute the quantity of food consumed
        food_quantity_df = self.compute_quantity_of_food(
            population_df, update_diet_df)

        # compute the surface needed in m^2
        food_surface_df_before = self.compute_surface(
            food_quantity_df, self.kg_to_m2_dict, population_df)
        
        self.food_surface_df_without_climate_change = food_surface_df_before
        #Add climate change impact to land required 
        surface_df = self.add_climate_impact(food_surface_df_before, temperature_df)
        # add years data
        surface_df['years'] = self.years

        self.food_land_surface_df = surface_df
        self.total_food_land_surface['years'] = surface_df['years']
        self.total_food_land_surface['total surface (Gha)'] = surface_df['total surface (Gha)']

        self.food_land_surface_percentage_df = self.convert_surface_to_percentage(
            surface_df)

        # compute cost details & price
        self.compute_price()
        # compute prod from invests
        self.compute_primary_energy_production()
        crop_energy_production = deepcopy(
            self.production['biomass_dry (TWh)'])
        # production of residue is the production from food surface and from
        # crop energy
        self.residue_prod_from_food_surface = self.compute_residue_from_food()
        self.mix_detailed_production['Crop residues (TWh)'] = self.residue_prod_from_food_surface.values +  \
            self.techno_infos_dict['residue_density_percentage'] * crop_energy_production

        self.mix_detailed_production['Crop for Energy (TWh)'] = crop_energy_production * (1 - self.techno_infos_dict['residue_density_percentage'])

        self.mix_detailed_production['Total (TWh)'] = self.mix_detailed_production['Crop for Energy (TWh)'] + self.mix_detailed_production['Crop residues (TWh)']

        # compute crop for energy land use
        self.compute_land_use()

        # no emisions
        self.CO2_emissions['Crop'] = np.zeros(len(self.years))

        # no consumption
        self.techno_consumption[f'{BiomassDry.name}'] = np.zeros(len(self.years))
        self.techno_consumption_woratio[f'{BiomassDry.name}'] = np.zeros(len(self.years))

    def compute_quantity_of_food(self, population_df, diet_df):
        """
        Compute the quantity of each food of the diet eaten each year

        @param population_df: input, give the population of each year
        @type population_df: dataframe
        @unit population_df: millions of people

        @param diet_df: input, contain the amount of food consumed each year, for each food considered
        @type diet_df: dataframe
        @unit doet_df: kg / person / year

        @param result: amount of food consumed by the global population, eahc year, for each food considered
        @type result: dataframe
        @unit result: kg / year
        """
        result = pd.DataFrame()
        for key in diet_df.keys():
            if key == "years":
                result[key] = diet_df[key]
            else:
                result[key] = population_df['population'].values * diet_df[key].values * 1e6
        # as population is in million of habitants, *1e6 is needed
        return(result)

    def compute_surface(self, quantity_of_food_df, kg_food_to_surface, population_df):
        """
        Compute the surface needed to produce a certain amount of food

        @param quantity_of_food_df: amount of food consumed by the global population, eahc year, for each food considered
        @type quantity_of_food_df: dataframe
        @unit quantity_of_food_df: kg / year

        @param kg_food_to_surface: input, the surface needed to produce 1kg of the considered food
        @type kg_food_to_surface: dict
        @unit kg_food_to_surface: m^2 / kg

        @param population_df: input, give the population of each year
        @type population_df: dataframe
        @unit population_df: millions of people

        @param result: the surface needed to produce the food quantity in input
        @type result: dataframe
        @unit result: m^2
        """
        result = pd.DataFrame()
        sum = np.array(len(quantity_of_food_df.index) * [0])
        for key in quantity_of_food_df.keys():
            if key == "years":
                pass
            else:
                result[key + ' (Gha)'] = kg_food_to_surface[key] * \
                    quantity_of_food_df[key]
                sum = sum + result[key + ' (Gha)']
        # add other contribution. 1e6 is for million of people,
        # /hatom2 for future conversion
        result['other (Gha)'] = self.other_use_crop * \
            population_df['population'].values * 1e6 / self.hatom2
        sum = sum + result['other (Gha)']

        # add total data
        result['total surface (Gha)'] = sum

        # put data in [Gha]
        result = result * self.hatom2 / 1e9

        return(result)

    def diet_change(self, diet_df, red_to_white_meat, meat_to_vegetables):
        """
        update diet data

        @param diet_df: diet of the first year of the study (first year only)
        @type diet_df: dataframe
        @unit diet_df: kg/person/year

        @param red_to_white_meat: percentage of the red meat "transformed" into white meat, regarding base diet
        @type diet_df: array
        @unit diet_df: %

        @param meat_to_vegetables:  percentage of the white meat "transformed" into fruit and vegetables, regarding base diet
        @type diet_df: array
        @unit diet_df: %

        @param result: changed_diet_df : contain the updated data of the quantity of each food, for each year of the study
        @param result: dataframe
        @unit result: kg/person/year
        """
        unity_array = np.array([1.] * len(red_to_white_meat))
        changed_diet_df = pd.DataFrame()
        # compute the number of kcal of red meat to transfert to white meat
        # /100 because red_to_white_meat is in %
        removed_red_meat = diet_df['red meat'].values[0] * \
            red_to_white_meat / 100

        # compute the kg of white meat needed to fill the kcal left by red meat
        added_white_meat = removed_red_meat * \
            self.kg_to_kcal_dict['red meat'] / \
            self.kg_to_kcal_dict['white meat']

        # update diet_df with new meat data
        for key in diet_df:
            if key == 'red meat':
                changed_diet_df[key] = diet_df[key].values[0] * \
                    (unity_array - red_to_white_meat / 100)
            elif key == 'white meat':
                changed_diet_df[key] = diet_df[key].values[0] * \
                    unity_array + added_white_meat
            else:
                changed_diet_df[key] = diet_df[key].values[0] * \
                    unity_array

        # compute the number of kcal of white meat to transfert to vegetables
        # /100 because meat_to_vegetables is in %
        removed_white_meat = changed_diet_df['white meat'].values * \
            meat_to_vegetables / 100 

        # compute the kg of fruit and vegetables needed to fill the kcal
        added_fruit = removed_white_meat * self.kg_to_kcal_dict['white meat'] / \
            self.kg_to_kcal_dict['fruits and vegetables']

        # update diet_df with new vegetables data
        changed_diet_df['white meat'] = changed_diet_df['white meat'].values - \
            removed_white_meat
        changed_diet_df['fruits and vegetables'] = changed_diet_df['fruits and vegetables'] + added_fruit

        return(changed_diet_df)

    def convert_surface_to_percentage(self, surface_df):
        """
        Express the surface taken by each food type in % and not in Gha

        @param surface_df: dataframe with the surface of each type of food expressed in Gha, and a columun with the total surface taken in Gha.
        @type surface_df: dataframe
        @unit surface_df: Gha

        @param land_surface_percentage_df: output of the method, with the share of surface taken by each type of food, in %
        @type land_surface_percentage_df: dataframe
        @unit land_surface_percentage_df: %
        """
        land_surface_percentage_df = deepcopy(surface_df)
        for key in land_surface_percentage_df.keys():
            if key == 'years':
                pass
            else:
                land_surface_percentage_df[key] = land_surface_percentage_df[key] / \
                    land_surface_percentage_df['total surface (Gha)'].values * 100

        land_surface_percentage_df.rename(
            columns=self.column_dict, inplace=True)
        return(land_surface_percentage_df)
    
    def add_climate_impact(self, surface_df_before, temperature_df):
        """ Add productivity reduction due to temperature increase and compute the new required surface
        Inputs: - surface_df_before: dataframe, Gha ,land required for food production without climate change
                - parameters of productivity function
                - temperature_df: dataframe, degree celsius wrt preindustrial level, dataframe of temperature increase
        """
        temperature = temperature_df['temp_atmo'].values
        #Compute the difference in temperature wrt 2020 reference
        temp = temperature - temperature_df.at[self.year_start, 'temp_atmo']
        #Compute reduction in productivity due to increase in temperature 
        pdctivity_reduction = self.param_a * temp**2 + self.param_b * temp
        self.prod_reduction = pdctivity_reduction
        self.productivity_evolution = pd.DataFrame({"years": self.years, 'productivity_evolution': pdctivity_reduction})
        #Apply this reduction to increase land surface needed
        surface_df = surface_df_before.multiply(other = (1- pdctivity_reduction), axis=0)
        
        return surface_df

    def updated_diet_expression(self, column_name):
        """
        gives the expression of the updated diet for each column.
        if  column is red meat : 
         updated_red_meat = diet_df[red meat] * (1 - convert_to_white_meat_factor)
         updated_white_meat = diet_df[white_meat] + added_WM - removed_WM


        @param column_name: name of the column
        @type column_name: string
        @unit column_name: none

        @param result: expression of the updated diet with the input data
        @type result: array
        @unit result: kg/person/year
        """
        unity_array = np.array([1.] * (self.year_end - self.year_start + 1))
        if column_name == "red meat":
            result = self.diet_df[column_name].values[0] * \
                (unity_array - self.red_to_white_meat / 100)

        elif column_name == "white meat":
            removed_red_meat = self.diet_df['red meat'].values[0] * \
                self.red_to_white_meat / 100
            added_white_meat = removed_red_meat * \
                self.kg_to_kcal_dict['red meat'] / \
                self.kg_to_kcal_dict['white meat']
            removed_white_meat = (self.diet_df["white meat"].values[0] * unity_array + added_white_meat) * \
                self.meat_to_vegetables / 100
            result = self.diet_df[column_name].values[0] * \
                unity_array + added_white_meat - removed_white_meat

        elif column_name == 'fruits and vegetables':
            removed_red_meat = self.diet_df['red meat'].values[0] * \
                self.red_to_white_meat / 100
            added_white_meat = removed_red_meat * \
                self.kg_to_kcal_dict['red meat'] / \
                self.kg_to_kcal_dict['white meat']
            removed_white_meat = (self.diet_df['white meat'].values[0] * unity_array + added_white_meat) * \
                self.meat_to_vegetables / 100
            added_fruits_and_vegetables = removed_white_meat * self.kg_to_kcal_dict['white meat'] / \
                self.kg_to_kcal_dict['fruits and vegetables']
            result = self.diet_df[column_name].values[0] * \
                unity_array + added_fruits_and_vegetables

        else:
            result = self.diet_df[column_name].values[0] * unity_array

        return(result)
    

    def compute_price(self):
        """
        Compute the cost details for crop & price
        """
        # Gather invests in crop for energy input
        invest_inputs = self.invest_level.loc[self.invest_level['years']
                                              <= self.cost_details['years'].max()]['invest'].values

        # Maximize with smooth exponential
        #this invest is not used in the price computation
        self.cost_details['invest'] = compute_func_with_exp_min(
            invest_inputs, self.min_value_invest)

        # CAPEX 
        capex_init = self.check_capex_unity(self.techno_infos_dict)
        self.cost_details['Capex ($/MWh)'] = capex_init * np.ones(len(self.cost_details['invest']))

        # CRF
        self.crf = self.compute_crf(self.techno_infos_dict)

        # Energy costs
        self.cost_details['Energy costs ($/MWh)'] = self.compute_other_primary_energy_costs()

        # Factory cost including CAPEX OPEX
        self.cost_details['Factory ($/MWh)'] = self.cost_details[f'Capex ($/MWh)'] * (self.crf + self.techno_infos_dict['Opex_percentage'])

        if 'nb_years_amort_capex' in self.techno_infos_dict:
            self.nb_years_amort_capex = self.techno_infos_dict['nb_years_amort_capex']

        # pylint: disable=no-member
        len_y = max(self.cost_details['years']) + \
            1 - min(self.cost_details['years'])
        self.cost_details['Crop_factory_amort'] = (np.tril(np.triu(np.ones((len_y, len_y)), k=0), k=self.nb_years_amort_capex - 1).transpose() *
                                                           np.array(self.cost_details['Factory ($/MWh)'].values / self.nb_years_amort_capex)).T.sum(axis=0)
        # pylint: enable=no-member

        # Compute and add transport
        self.cost_details['Transport ($/MWh)'] = self.transport_cost['transport'] * \
            self.transport_margin['margin'] / 100.0 / self.data_fuel_dict['calorific_value']

        # Crop amort
        self.cost_details['Crop amort ($/MWh)'] = self.cost_details['Crop_factory_amort'] + self.cost_details['Transport ($/MWh)'] + \
            self.cost_details['Energy costs ($/MWh)']

        # Total cost (MWh)
        self.cost_details['Total ($/MWh)'] = self.cost_details['Energy costs ($/MWh)'] +  self.cost_details['Factory ($/MWh)'] + self.cost_details['Transport ($/MWh)']

        # Add margin in %
        self.cost_details['Total ($/MWh)'] *= self.margin.loc[self.margin['years']
                                                        <= self.cost_details['years'].max()]['margin'].values / 100.0                                  
        self.cost_details['Crop amort ($/MWh)'] *= self.margin.loc[self.margin['years']
                                                                   <= self.cost_details['years'].max()]['margin'].values / 100.0
        # Total cost (t)
        self.cost_details['Total ($/t)'] = self.cost_details['Total ($/MWh)'] * self.data_fuel_dict['calorific_value']
        self.techno_prices['Crop'] = self.cost_details['Total ($/MWh)']

        if 'CO2_taxes_factory' in self.cost_details:
            self.techno_prices['Crop_wotaxes'] = self.cost_details['Total ($/MWh)'] - \
                self.cost_details['CO2_taxes_factory']
        else:
            self.techno_prices['Crop_wotaxes'] = self.cost_details['Total ($/MWh)']

        price_crop = self.cost_details['Total ($/t)'] / \
            (1 + self.techno_infos_dict['residue_density_percentage'] *
             (self.techno_infos_dict['crop_residue_price_percent_dif'] - 1))

        price_residue = price_crop * \
            self.techno_infos_dict['crop_residue_price_percent_dif']

        # Price_residue = crop_residue_ratio * Price_crop
        # Price_crop = Price_tot / ((1-ratio_prices)*crop_residue_ratio + ratio_prices)
        self.mix_detailed_prices['Price crop ($/t)'] = price_crop
        self.mix_detailed_prices['Price residue ($/t)'] = price_residue

    def check_capex_unity(self, data_tocheck):
        """
        Put all capex in $/MWh
        """
        if data_tocheck['Capex_init_unit'] == 'euro/ha':

            density_per_ha = data_tocheck['density_per_ha']

            if data_tocheck['density_per_ha_unit'] == 'm^3/ha':
                capex_init = data_tocheck['Capex_init'] * \
                    data_tocheck['euro_dollar'] / \
                    density_per_ha / \
                    data_tocheck['density'] / \
                    self.data_fuel_dict['calorific_value']

            elif data_tocheck['density_per_ha_unit'] == 'kg/ha':
                capex_init = data_tocheck['Capex_init'] * \
                    data_tocheck['euro_dollar'] / \
                    density_per_ha / \
                    self.data_fuel_dict['calorific_value']
        else:
            capex_unit = data_tocheck['Capex_init_unit']
            raise Exception(
                f'The CAPEX unity {capex_unit} is not handled yet in techno_type')

        # return capex in $/MWh
        return capex_init * 1.0e3

    def compute_other_primary_energy_costs(self):
        """
        Compute primary costs to produce 1 MWh of crop
        """
        
        return 0.0

    def compute_crf(self, data_config):
        """
        Compute annuity factor with the Weighted averaged cost of capital
        and the lifetime of the selected solution
        """
        crf = (data_config['WACC'] * (1.0 + data_config['WACC']) ** data_config['lifetime']) / \
              ((1.0 + data_config['WACC']) ** data_config['lifetime'] - 1.0)

        return crf

    def compute_primary_energy_production(self):
        '''
        Compute biomass_dry production
        '''
        # First compute the initial aging distribution with the initial
        # production found in the techno discipline
        # self.compute_initial_aging_distribution()

        # Compute the aging distribution over the years of study to determine the total production over the years
        # This function also erase old factories from the distribution
        self.compute_aging_distribution_production()

        # Finally compute the production by summing all aged production for
        # each year

        age_distrib_prod_sum = self.age_distrib_prod_df.groupby(['years'], as_index=False).agg({f'distrib_prod (TWh)': 'sum'}
                                                                                               )
        if 'biomass_dry (TWh)' in self.production:
            del self.production['biomass_dry (TWh)']

        self.production = pd.merge(self.production, age_distrib_prod_sum, how='left', on='years').rename(
            columns={'distrib_prod (TWh)': 'biomass_dry (TWh)'}).fillna(0.0)
        

    def compute_aging_distribution_production(self):
        '''
        Compute the aging distribution production of primary energy for years of study
        Start with the initial distribution and add a year on the age each year 
        Add also the yearly production regarding the investment
        All productions older than the lifetime are removed from the dataframe  
        '''
        # To break the object link with initial distrib
        aging_distrib_year_df = pd.DataFrame(
            {'age': self.initial_age_distrib['age'].values})
        aging_distrib_year_df[f'distrib_prod (TWh)'] = self.initial_age_distrib['distrib'] * \
            self.initial_production / 100.0

        production_from_invest = self.compute_prod_from_invest(
            construction_delay=self.construction_delay)

        # get the whole dataframe for new production with one line for each
        # year at each age
        # concatenate tuple to get correct df to mimic the old year loop
        len_years = len(self.years)
        range_years = np.arange(
            self.year_start, self.year_end + len_years)

        year_array = np.concatenate(
            tuple(range_years[i:i + len_years] for i in range(len_years)))
        age_array = np.concatenate(tuple(np.ones(
            len_years) * (len_years - i) for i in range(len_years, 0, -1)))
        prod_array = production_from_invest['prod_from_invest'].values.tolist(
        ) * len_years

        new_prod_aged = pd.DataFrame({'years': year_array, 'age': age_array, 'distrib_prod (TWh)': prod_array})

        # get the whole dataframe for old production with one line for each
        # year at each age
        year_array = np.array([[year] * len(aging_distrib_year_df)
                               for year in self.years]).flatten()
        age_values = aging_distrib_year_df['age'].values
        age_array = np.concatenate(tuple(
            age_values + i for i in range(len_years)))
        prod_array = aging_distrib_year_df[f'distrib_prod (TWh)'].values.tolist(
        ) * len_years

        old_prod_aged = pd.DataFrame({'years': year_array, 'age': age_array,
                                      f'distrib_prod (TWh)': prod_array})

        # Concat the two created df
        self.age_distrib_prod_df = pd.concat(
            [new_prod_aged, old_prod_aged], ignore_index=True)

        self.age_distrib_prod_df = self.age_distrib_prod_df.loc[
            # Suppress all lines where age is higher than lifetime
            (self.age_distrib_prod_df['age'] <
             self.techno_infos_dict['lifetime'])
            # Suppress all lines where age is higher than lifetime
            & (self.age_distrib_prod_df['years'] < self.year_end + 1)
            # Fill Nan with zeros and suppress all zeros
            & (self.age_distrib_prod_df[f'distrib_prod (TWh)'] != 0.0)
        ]
        # Fill Nan with zeros
        self.age_distrib_prod_df.fillna(0.0, inplace=True)

    def compute_prod_from_invest(self, construction_delay):
        '''
        Compute the crop production from investment in t
        '''
        prod_before_ystart = pd.DataFrame({'years': np.arange(self.year_start - construction_delay, self.year_start),
                                           'invest': [0.0]*(construction_delay),
                                           'Capex ($/MWh)': self.cost_details.loc[self.cost_details[
                                           'years'] == self.year_start, 'Capex ($/MWh)'].values[0]})

        production_from_invest = pd.concat(
            [self.cost_details[['years', 'invest', 'Capex ($/MWh)']], prod_before_ystart], ignore_index=True)

        production_from_invest.sort_values(by=['years'], inplace=True)
        ## Invest in M$ | Capex in $/MWh | Prod in TWh
        production_from_invest['prod_from_invest'] = production_from_invest['invest'].values / \
            production_from_invest[f'Capex ($/MWh)'].values

        production_from_invest['years'] += construction_delay

        production_from_invest = production_from_invest[production_from_invest['years']
                                                        <= self.year_end]
        return production_from_invest
        
    def compute_residue_from_food(self):
        '''
        Compute residue part from the land surface for food.
        '''
        # compute residue part from food land surface for energy sector in TWh
        residue_production = self.total_food_land_surface['total surface (Gha)'] * \
            self.techno_infos_dict['residue_density_percentage'] * \
            self.techno_infos_dict['density_per_ha'] * \
            self.data_fuel_dict['calorific_value'] * \
            self.techno_infos_dict['residue_percentage_for_energy']

        return residue_production

    def compute_land_use(self):
        """
        Compute land use required for crop for energy
        """
        self.land_use_required['Crop for Energy (Gha)'] = self.mix_detailed_production['Crop for Energy (TWh)'] / \
                                                       self.techno_infos_dict['density_per_ha']
        self.land_use_required['Crop for Food (Gha)'] = self.food_land_surface_df['total surface (Gha)']


    ####### Gradient #########

    def d_land_surface_d_population(self, column_name):
        """
        Compute the derivate of food_land_surface_df wrt population, for a specific column.
        derivate_step1 = diet_expression(kg/person/year) * kg_to_m2 / 1e7
        derivative = derivative_step1 * (1- productivity reduction due to temperature increase)
        diet_expression is computed in a separated method

        need self.column_dict because input column get '(Gha)' at the end
        / 1e7 comes from the unit : *1e6 (population in million) /1e4 (m2 to ha) /1e9 (ha to Gha)
        """
        updated_diet_food = self.updated_diet_expression(
            self.column_dict[column_name])
        number_of_values = (self.year_end - self.year_start + 1)
        d_land_surface_d_pop_before = np.identity(
            number_of_values) * updated_diet_food * self.kg_to_m2_dict[self.column_dict[column_name]] / 1e7
        #Add climate change impact 
        d_land_surface_d_pop = d_land_surface_d_pop_before * (1 - self.prod_reduction)
        
        return(d_land_surface_d_pop)

    def d_other_surface_d_population(self):
        """
        Compute derivate of land_surface[other] column wrt population_df[population]
        """
        number_of_values = (self.year_end - self.year_start + 1)
        result_without_climate = np.identity(
            number_of_values) * self.other_use_crop / 1e3
        #Add climate change impact
        result = result_without_climate * (1 - self.prod_reduction)
        
        return(result)
    
    def d_food_land_surface_d_temperature(self, temperature_df, column_name):
        """
        Compute the derivative of land surface wrt temperature
        Final land surface = food_land_surface_step_one * (1 - productivity)
        productivity = f(temperature)
        d_food_land_surface_d_temperature =  d_land_d_productivity * d_productivity_d_temperature
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        temp_zero = temperature_df.at[self.year_start, 'temp_atmo']
        temp = temperature_df['temp_atmo'].values
        a = self.param_a
        b = self.param_b
        land_before = self.food_surface_df_without_climate_change[column_name].values
        #Step 1: Productivity reduction
        #temp = temperature - temperature_df.at[self.year_start, 'temp_atmo']
        #pdctivity_reduction = self.param_a * temp**2 + self.param_b * temp
        #=at**2 + at0**2 - 2att0 + bt - bt0
        #Derivative wrt t each year:  2at-2at0 +b
        d_productivity_d_temperature = idty * (2*a*temp - 2*a*temp_zero + b)
        #Add derivative wrt t0: 2at0 -2at -b
        d_productivity_d_temperature[:, 0] += 2*a*temp_zero - 2*a*temp - b
        #Step 2:d_climate_d_productivity for each t: land = land_before * (1 - productivity) 
        d_land_d_productivity = - idty * land_before
        d_food_land_surface_d_temperature = d_land_d_productivity.dot(d_productivity_d_temperature)
        
        return d_food_land_surface_d_temperature

    def d_surface_d_red_to_white(self, population_df):
        """
        Compute the derivative of total food land surface wrt red to white percentage design variable
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        kg_food_to_surface = self.kg_to_m2_dict
        #red to white meat value influences red meat, white meat, and vegetable surface
        red_meat_diet_grad = - self.diet_df['red meat'].values[0] / 100 
        white_meat_diet_grad = self.diet_df['red meat'].values[0] / 100 * self.kg_to_kcal_dict['red meat'] / self.kg_to_kcal_dict['white meat'] \
            * (1 - self.meat_to_vegetables / 100)
        fruits_diet_grad = self.diet_df['red meat'].values[0] / 100 * self.kg_to_kcal_dict['red meat'] / self.kg_to_kcal_dict['white meat'] \
            * self.meat_to_vegetables / 100 * self.kg_to_kcal_dict['white meat'] / self.kg_to_kcal_dict['fruits and vegetables']
        
        red_meat_quantity_grad = red_meat_diet_grad * population_df['population'].values * 1e6
        white_meat_quantity_grad = white_meat_diet_grad * population_df['population'].values * 1e6
        fruits_quantity_grad = fruits_diet_grad * population_df['population'].values * 1e6

        red_meat_surface_grad = red_meat_quantity_grad * kg_food_to_surface['red meat'] * self.hatom2 / 1e9
        white_meat_surface_grad = white_meat_quantity_grad * kg_food_to_surface['white meat'] * self.hatom2 / 1e9
        fruits_meat_quantity_grad = fruits_quantity_grad * kg_food_to_surface['fruits and vegetables'] * self.hatom2 / 1e9

        total_surface_grad = red_meat_surface_grad + white_meat_surface_grad + fruits_meat_quantity_grad
        total_surface_climate_grad = total_surface_grad * (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad.values * idty

    def d_surface_d_meat_to_vegetable(self, population_df):
        """
        Compute the derivative of total food land surface wrt meat to vegetable percentage design variable
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        kg_food_to_surface = self.kg_to_m2_dict
        #meat to vegetables value influences white meat and vegetable surface 
        white_meat_diet_grad = - (self.diet_df['white meat'].values[0] + self.diet_df['red meat'].values[0] * \
            self.red_to_white_meat / 100 * self.kg_to_kcal_dict['red meat'] / self.kg_to_kcal_dict['white meat']) / 100
        fruits_diet_grad = - white_meat_diet_grad * self.kg_to_kcal_dict['white meat'] / self.kg_to_kcal_dict['fruits and vegetables']
        
        white_meat_quantity_grad = white_meat_diet_grad * population_df['population'].values * 1e6
        fruits_quantity_grad = fruits_diet_grad * population_df['population'].values * 1e6

        white_meat_surface_grad = white_meat_quantity_grad * kg_food_to_surface['white meat'] * self.hatom2 / 1e9
        fruits_meat_quantity_grad = fruits_quantity_grad * kg_food_to_surface['fruits and vegetables'] * self.hatom2 / 1e9

        total_surface_grad = white_meat_surface_grad + fruits_meat_quantity_grad
        total_surface_climate_grad = total_surface_grad * (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad.values * idty

    def compute_dprod_from_dinvest(self):
        #return dproduction_from_dinvest

        '''
               Compute the partial derivative of prod vs invest  and the partial derivative of prod vs capex
               To compute after the total derivative of prod vs invest = dpprod_dpinvest + dpprod_dpcapex*dcapexdinvest
               with dcapexdinvest already computed for detailed prices
               '''

        nb_years = (self.year_end - self.year_start + 1)
        arr_type = 'float64'
        dprod_list_dinvest_list = np.zeros(
            (nb_years, nb_years), dtype=arr_type)

        # We fill this jacobian column by column because it is the same element
        # in the entire column
        for i in range(nb_years):

            len_non_zeros = min(max(0, nb_years - self.construction_delay - i), self.techno_infos_dict['lifetime'])
            first_len_zeros = min( i + self.construction_delay, nb_years)
            last_len_zeros = max(0, nb_years - len_non_zeros - first_len_zeros)
            # For prod in each column there is lifetime times the same value which is dpprod_dpinvest
            # This value is delayed in time (means delayed in lines for
            # jacobian by construction _delay)
            # Each column is then composed of [0,0,0... (dp/dx,dp/dx)*lifetime,
            # 0,0,0]
            dpprod_dpinvest = 1 / self.cost_details[f'Capex ($/MWh)'].values[i] /\
                                                             self.data_fuel_dict['calorific_value']
            is_invest_negative = max(
                np.sign(self.cost_details['invest'].values[i] + np.finfo(float).eps), 0.0)
            dprod_list_dinvest_list[:, i] = np.hstack((np.zeros(first_len_zeros),
                                                       np.ones(
                                                           len_non_zeros) * dpprod_dpinvest * is_invest_negative,
                                                       np.zeros(last_len_zeros)))

        # Mt to GWh
        return dprod_list_dinvest_list

    def compute_d_prod_dland_for_food(self, dland_for_food):
        '''
        Compute gradient of production from land surface from food
        an identity matrice with the same scalar on diagonal

        '''
        # production = residue production from food + crop energy production
        # residue production from food = compute_residue_from_food_investment
        d_prod_dland_for_food = dland_for_food * \
                                self.techno_infos_dict['residue_density_percentage'] * \
                                self.techno_infos_dict['density_per_ha'] * \
                                self.techno_infos_dict['residue_percentage_for_energy'] * \
                                self.data_fuel_dict['calorific_value'] / \
                                self.scaling_factor_techno_production
        return d_prod_dland_for_food
