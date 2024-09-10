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

from climateeconomics.glossarycore import GlossaryCore


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


class Agriculture():
    """
    Agriculture pyworld3 class

    basic for now, to evolve 

    """

    KM_2_unit = 'km2'
    HECTARE = 'ha'

    YEAR_START = GlossaryCore.YearStart
    YEAR_END = GlossaryCore.YearEnd
    TIME_STEP = GlossaryCore.TimeStep
    POPULATION_DF = GlossaryCore.PopulationDfValue
    DIET_DF = 'diet_df'
    KG_TO_KCAL_DICT = 'kg_to_kcal_dict'
    KG_TO_M2_DICT = 'kg_to_m2_dict'
    OTHER_USE_AGRICULTURE = 'other_use_agriculture'

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
        self.year_start = self.param[Agriculture.YEAR_START]
        self.year_end = self.param[Agriculture.YEAR_END]
        self.time_step = self.param[Agriculture.TIME_STEP]
        self.diet_df = self.param[Agriculture.DIET_DF]
        self.kg_to_kcal_dict = self.param[Agriculture.KG_TO_KCAL_DICT]
        self.kg_to_m2_dict = self.param[Agriculture.KG_TO_M2_DICT]
#         self.red_meat_percentage = self.param['red_meat_percentage']['red_meat_percentage']
#         self.white_meat_percentage = self.param['white_meat_percentage']['white_meat_percentage']
        self.other_use_agriculture = self.param[Agriculture.OTHER_USE_AGRICULTURE]
        self.param_a = self.param['param_a']
        self.param_b = self.param['param_b']

    def apply_percentage(self, inp_dict):

        self.red_meat_percentage = inp_dict['red_meat_percentage']['red_meat_percentage'].values
        self.white_meat_percentage = inp_dict['white_meat_percentage']['white_meat_percentage'].values

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

        # Set index of coupling dataframe in inputs
        temperature_df.index = temperature_df[GlossaryCore.Years].values

        # construct the diet over time
        self.updated_diet_df = deepcopy(self.update_diet())

        # compute the quantity of food consumed
        food_quantity_df = self.compute_quantity_of_food(
            population_df, self.updated_diet_df)

        # compute the surface needed in m^2
        food_surface_df_before = self.compute_surface(
            food_quantity_df, self.kg_to_m2_dict, population_df)

        self.food_surface_df_without_climate_change = food_surface_df_before
        # Add climate change impact to land required
        surface_df = self.add_climate_impact(
            food_surface_df_before, temperature_df)
        # add years data
        surface_df.insert(loc=0, column=GlossaryCore.Years, value=self.years)

        self.food_land_surface_df = surface_df

        self.total_food_land_surface[GlossaryCore.Years] = surface_df[GlossaryCore.Years]
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
            if key == GlossaryCore.Years:
                result[key] = diet_df[key]
            else:
                result[key] = population_df[GlossaryCore.PopulationValue] * diet_df[key] * 1e6
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
            if key == GlossaryCore.Years:
                pass
            else:
                result[key + ' (Gha)'] = kg_food_to_surface[key] * \
                    quantity_of_food_df[key]
                sum = sum + result[key + ' (Gha)']
        # add other contribution. 1e6 is for million of people,
        # /hatom2 for future conversion
        result['other (Gha)'] = self.other_use_agriculture * \
            population_df[GlossaryCore.PopulationValue].values * 1e6 / self.hatom2
        sum = sum + result['other (Gha)']

        # add total data
        result['total surface (Gha)'] = sum

        # put data in [Gha]
        result = result * self.hatom2 / 1e9

        return(result)

    def update_diet(self):
        '''
            update diet data:
                - compute new kcal/person/year from red and white meat
                - update proportionally all vegetable kcal/person/year
                - compute new diet_df
        '''
        starting_diet = self.diet_df
        changed_diet_df = pd.DataFrame({GlossaryCore.Years: self.years})
        self.kcal_diet_df = {}
        total_kcal = 0
        # compute total kcal
        for key in starting_diet:
            self.kcal_diet_df[key] = starting_diet[key].values[0] * \
                self.kg_to_kcal_dict[key]
            total_kcal += self.kcal_diet_df[key]
        self.kcal_diet_df['total'] = total_kcal

        # compute the kcal changed of red meat:
        # kg_food/person/year
        changed_diet_df['red meat'] = self.kcal_diet_df['total'] * \
            self.red_meat_percentage / 100 / self.kg_to_kcal_dict['red meat']
        changed_diet_df['white meat'] = self.kcal_diet_df['total'] * \
            self.white_meat_percentage / 100 / \
            self.kg_to_kcal_dict['white meat']

        # removed kcal/person/year
        removed_red_meat_kcal = self.kcal_diet_df['red meat'] - \
            self.kcal_diet_df['total'] * self.red_meat_percentage / 100
        removed_white_meat_kcal = self.kcal_diet_df['white meat'] - \
            self.kcal_diet_df['total'] * self.white_meat_percentage / 100

        for key in starting_diet:
            # compute new vegetable diet in kg_food/person/year: add the
            # removed_kcal/3 for each 3 category of vegetable
            if key == 'fruits and vegetables' or key == 'potatoes' or key == 'rice and maize':
                proportion = self.kcal_diet_df[key] / \
                    (self.kcal_diet_df['fruits and vegetables'] + 
                     self.kcal_diet_df['potatoes'] + self.kcal_diet_df['rice and maize'])
                changed_diet_df[key] = [starting_diet[key].values[0]] * len(self.years) + \
                    (removed_red_meat_kcal + removed_white_meat_kcal) * \
                    proportion / self.kg_to_kcal_dict[key]
            # no impact on eggs and milk
            elif key != 'red meat' and key != 'white meat':
                changed_diet_df[key] = [
                    starting_diet[key].values[0]] * len(self.years)

        return changed_diet_df

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
            if key == GlossaryCore.Years:
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
        temperature = temperature_df[GlossaryCore.TempAtmo]
        # Compute the difference in temperature wrt 2020 reference
        temp = temperature - temperature_df.at[self.year_start, GlossaryCore.TempAtmo]
        # Compute reduction in productivity due to increase in temperature
        pdctivity_reduction = self.param_a * temp ** 2 + self.param_b * temp
        self.prod_reduction = pdctivity_reduction
        self.productivity_evolution = pd.DataFrame(
            {GlossaryCore.Years: self.years, 'productivity_evolution': pdctivity_reduction})
        self.productivity_evolution.index = self.years
        # Apply this reduction to increase land surface needed
        surface_df = surface_df_before.multiply(
            other=(1 - pdctivity_reduction.values), axis=0)

        return surface_df

    ####### Gradient #########

    def d_land_surface_d_population(self, column_name_Gha):
        """
        Compute the derivate of food_land_surface_df wrt population, for a specific column.
        derivate_step1 = diet_expression(kg/person/year) * kg_to_m2 / 1e7
        derivative = derivative_step1 * (1- productivity reduction due to temperature increase)
        diet_expression is computed in a separated method

        need self.column_dict because input column get '(Gha)' at the end
        / 1e7 comes from the unit : *1e6 (population in million) /1e4 (m2 to ha) /1e9 (ha to Gha)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        column_name = self.column_dict[column_name_Gha]
        d_land_surface_d_pop_before = np.identity(
            number_of_values) * self.updated_diet_df[column_name].values * self.kg_to_m2_dict[column_name] / 1e7
        # Add climate change impact
        d_land_surface_d_pop = d_land_surface_d_pop_before * \
            (1 - self.prod_reduction.values)

        return(d_land_surface_d_pop)

    def d_other_surface_d_population(self):
        """
        Compute derivate of land_surface[other] column wrt population_df[population]
        """
        number_of_values = (self.year_end - self.year_start + 1)
        result_without_climate = np.identity(
            number_of_values) * self.other_use_agriculture / 1e3
        # Add climate change impact
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
        temp_zero = temperature_df[GlossaryCore.TempAtmo].values[0]
        temp = temperature_df[GlossaryCore.TempAtmo].values
        a = self.param_a
        b = self.param_b
        land_before = self.food_surface_df_without_climate_change[column_name].values
        # Step 1: Productivity reduction
        # temp = temperature - temperature_df.at[self.year_start, GlossaryCore.TempAtmo]
        # pdctivity_reduction = self.param_a * temp**2 + self.param_b * temp
        # =at**2 + at0**2 - 2att0 + bt - bt0
        # Derivative wrt t each year:  2at-2at0 +b
        d_productivity_d_temperature = idty * \
            (2 * a * temp - 2 * a * temp_zero + b)
        # Add derivative wrt t0: 2at0 -2at -b
        d_productivity_d_temperature[:, 0] += 2 * \
            a * temp_zero - 2 * a * temp - b
        # Step 2:d_climate_d_productivity for each t: land = land_before * (1 -
        # productivity)
        d_land_d_productivity = -idty * land_before
        d_food_land_surface_d_temperature = d_land_d_productivity.dot(
            d_productivity_d_temperature)

        return d_food_land_surface_d_temperature

    def d_surface_d_red_meat_percentage(self, population_df):
        """
        Compute the derivative of total food land surface wrt red meat percentage design variable
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        kg_food_to_surface = self.kg_to_m2_dict
        # red to white meat value influences red meat, white meat, and
        # vegetable surface
        red_meat_diet_grad = self.kcal_diet_df['total'] / 100 / \
            self.kg_to_kcal_dict['red meat'] * kg_food_to_surface['red meat']
        removed_red_kcal = -self.kcal_diet_df['total'] / 100

        vegetables_column_names = [
            'fruits and vegetables', 'potatoes', 'rice and maize']
        # sub total gradient is the sum of all gradients of food category
        sub_total_surface_grad = red_meat_diet_grad
        for vegetable_name in vegetables_column_names:

            proportion = self.kcal_diet_df[vegetable_name] / \
                (self.kcal_diet_df['fruits and vegetables'] + 
                 self.kcal_diet_df['potatoes'] + self.kcal_diet_df['rice and maize'])
            sub_total_surface_grad = sub_total_surface_grad + removed_red_kcal * proportion / \
                self.kg_to_kcal_dict[vegetable_name] * \
                kg_food_to_surface[vegetable_name]

        total_surface_grad = sub_total_surface_grad * \
            population_df[GlossaryCore.PopulationValue].values * 1e6 * self.hatom2 / 1e9
        total_surface_climate_grad = total_surface_grad * \
            (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad.values * idty

    def d_surface_d_white_meat_percentage(self, population_df):
        """
        Compute the derivative of total food land surface wrt white meat percentage design variable
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        kg_food_to_surface = self.kg_to_m2_dict
        # red to white meat value influences red meat, white meat, and
        # vegetable surface
        white_meat_diet_grad = self.kcal_diet_df['total'] / 100 / \
            self.kg_to_kcal_dict['white meat'] * \
            kg_food_to_surface['white meat']
        removed_white_kcal = -self.kcal_diet_df['total'] / 100

        vegetables_column_names = [
            'fruits and vegetables', 'potatoes', 'rice and maize']
        # sub total gradient is the sum of all gradients of food category
        sub_total_surface_grad = white_meat_diet_grad
        for vegetable_name in vegetables_column_names:

            proportion = self.kcal_diet_df[vegetable_name] / \
                (self.kcal_diet_df['fruits and vegetables'] + 
                 self.kcal_diet_df['potatoes'] + self.kcal_diet_df['rice and maize'])
            sub_total_surface_grad = sub_total_surface_grad + removed_white_kcal * proportion / \
                self.kg_to_kcal_dict[vegetable_name] * \
                kg_food_to_surface[vegetable_name]

        total_surface_grad = sub_total_surface_grad * \
            population_df[GlossaryCore.PopulationValue].values * 1e6 * self.hatom2 / 1e9
        total_surface_climate_grad = total_surface_grad * \
            (1 - self.productivity_evolution['productivity_evolution'])

        return total_surface_climate_grad.values * idty
