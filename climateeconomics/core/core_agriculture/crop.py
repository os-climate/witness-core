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

    def __init__(self, param):
        '''
        Constructor
        '''
        self.param = param
        self.world_surface_data = None
        self.hatom2 = 0.0001
        self.m2toha = 10000.
        self.surface_df = None

        self.set_data()
        self.create_dataframe()

    def set_data(self):
        self.year_start = self.param[Crop.YEAR_START]
        self.year_end = self.param[Crop.YEAR_END]
        self.time_step = self.param[Crop.TIME_STEP]
        self.diet_df = self.param[Crop.DIET_DF]
        self.kg_to_kcal_dict = self.param[Crop.KG_TO_KCAL_DICT]
        self.kg_to_m2_dict = self.param[Crop.KG_TO_M2_DICT]
        self.red_to_white_meat = self.param[Crop.RED_TO_WHITE_MEAT]
        self.meat_to_vegetables = self.param[Crop.MEAT_TO_VEGETABLES]
        self.other_use_crop = self.param[Crop.OTHER_USE_CROP]
        self.param_a = self.param['param_a']
        self.param_b = self.param['param_b']

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        years = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years = years
        self.food_land_surface_df = pd.DataFrame()
        self.total_food_land_surface = pd.DataFrame()
        self.column_dict = {'red meat (Gha)': 'red meat', 'white meat (Gha)': 'white meat',
                            'milk (Gha)': 'milk', 'eggs (Gha)': 'eggs', 'rice and maize (Gha)': 'rice and maize',
                            'potatoes (Gha)': 'potatoes', 'fruits and vegetables (Gha)': 'fruits and vegetables',
                            'other (Gha)': 'other', 'total surface (Gha)': 'total surface'}

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
        
        #Set index of coupling dataframe in inputs
        temperature_df.index = temperature_df['years'].values
        
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

#         self.percentage_diet_df = self.convert_diet_kcal_to_percentage(
#             update_diet_df)

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
        population_df.index = diet_df.index
        for key in diet_df.keys():
            if key == "years":
                result[key] = diet_df[key]
            else:
                result[key] = population_df['population'] * diet_df[key] * 1e6
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

        changed_diet_df.index = np.arange(
            self.year_start, self.year_end + 1, 1)

        return(changed_diet_df)

#     def convert_diet_kcal_to_percentage(self, diet_df):
#         """
#         """
#         percentage_diet_df = diet_df
#         sum = np.array([0] * len(percentage_diet_df.index))
#
#         for key in self.kg_to_kcal_dict.keys():
#             percentage_diet_df[key] = percentage_diet_df[key] * \
#                 self.kg_to_kcal_dict[key]
#             sum = sum + percentage_diet_df[key]
#
#         percentage_diet_df = percentage_diet_df / sum
#
#         return(percentage_diet_df)

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
        temperature = temperature_df['temp_atmo']
        #Compute the difference in temperature wrt 2020 reference
        temp = temperature - temperature_df.at[self.year_start, 'temp_atmo']
        #Compute reduction in productivity due to increase in temperature 
        pdctivity_reduction = self.param_a * temp**2 + self.param_b * temp
        self.prod_reduction = pdctivity_reduction
        self.productivity_evolution = pd.DataFrame({"years": self.years, 'productivity_evolution': pdctivity_reduction})
        self.productivity_evolution.index = self.years
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
            result = self.diet_df[column_name
                                  ].values[0] * unity_array
        return(result)

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
        d_land_surface_d_pop = d_land_surface_d_pop_before * (1 - self.prod_reduction.values)
        
        return(d_land_surface_d_pop)

    def d_other_surface_d_population(self):
        """
        Compute derivate of land_surface[other] column wrt population_df[population]
        """
        number_of_values = (self.year_end - self.year_start + 1)
        result_without_climate = np.identity(
            number_of_values) * self.other_use_crop / 1e3
        #Add climate change impact
        result = result_without_climate * (1 - self.prod_reduction.values)
        
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
        
        red_meat_quantity_grad = red_meat_diet_grad * population_df['population'] * 1e6
        white_meat_quantity_grad = white_meat_diet_grad * population_df['population'] * 1e6
        fruits_quantity_grad = fruits_diet_grad * population_df['population'] * 1e6

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
        
        white_meat_quantity_grad = white_meat_diet_grad * population_df['population'] * 1e6
        fruits_quantity_grad = fruits_diet_grad * population_df['population'] * 1e6

        white_meat_surface_grad = white_meat_quantity_grad * kg_food_to_surface['white meat'] * self.hatom2 / 1e9
        fruits_meat_quantity_grad = fruits_quantity_grad * kg_food_to_surface['fruits and vegetables'] * self.hatom2 / 1e9

        total_surface_grad = white_meat_surface_grad + fruits_meat_quantity_grad
        total_surface_climate_grad = total_surface_grad * (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad.values * idty
