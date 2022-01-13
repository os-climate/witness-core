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


class LandUse():
    """
    Land use model class 

    basic for now, to evolve 

    source: https://ourworldindata.org/land-use
    """

    KM_2_unit = 'km2'
    HECTARE = 'ha'

    LAND_DEMAND_DF = 'land_demand_df'
    YEAR_START = 'year_start'
    YEAR_END = 'year_end'
    CROP_LAND_USE_PER_CAPITA = 'crop_land_use_per_capita'
    LIVESTOCK_LAND_USE_PER_CAPITA = 'livestock_land_use_per_capita'
    ECONOMICS_DF = 'economics_df'
    POPULATION_DF = 'population_df'
    LIVESTOCK_USAGE_FACTOR_DF = 'livestock_usage_factor_df'

    LAND_DEMAND_CONSTRAINT_DF = 'land_demand_constraint_df'
    LAND_DEMAND_CONSTRAINT_AGRICULTURE = 'Agriculture demand constraint (Gha)'
    LAND_DEMAND_CONSTRAINT_FOREST = 'Forest demand constraint (Gha)'

    LAND_SURFACE_DF = 'land_surface_df'
    LAND_SURFACE_DETAIL_DF = 'land_surface_detail_df'
    LAND_SURFACE_FOR_FOOD_DF = 'land_surface_for_food_df'

    LAND_USE_CONSTRAINT_REF = 'land_use_constraint_ref'
    # Technologies filteref by land type
    FOREST_TECHNO = ['ManagedWood (Gha)', 'UnmanagedWood (Gha)',
                     'Reforestation (Gha)']
    AGRICULTURE_TECHNO = [
        'CropEnergy (Gha)', 'SolarPv (Gha)', 'SolarThermal (Gha)']

    def __init__(self, param):
        '''
        Constructor
        '''
        self.param = param
        self.world_surface_data = None
        self.ha2km2 = 0.01
        self.km2toha = 100.
        self.surface_file = 'world_surface_data.csv'
        self.surface_df = None
        self.import_world_surface_data()

        self.set_data()
        self.land_demand_constraint_df = None
        self.land_surface_df = pd.DataFrame()
        self.land_surface_for_food_df = pd.DataFrame()

    def set_data(self):
        self.year_start = self.param[LandUse.YEAR_START]
        self.year_end = self.param[LandUse.YEAR_END]
        self.crop_land_use_per_capita = self.param[LandUse.CROP_LAND_USE_PER_CAPITA]
        self.livestock_land_use_per_capita = self.param[LandUse.LIVESTOCK_LAND_USE_PER_CAPITA]
        self.livestock_usage_factor_df = self.param[LandUse.LIVESTOCK_USAGE_FACTOR_DF]
        self.ref_land_use_constraint = self.param[LandUse.LAND_USE_CONSTRAINT_REF]

    def import_world_surface_data(self):
        curr_dir = os.path.dirname(__file__)
        data_file = os.path.join(curr_dir, self.surface_file)
        self.surface_df = pd.read_csv(data_file)

    def compute(self, population_df, land_demand_df, livestock_usage_factor_df):
        ''' 
        Computation methods, comput land demands and constraints

        @param population_df: population from input
        @type population_df: dataframe

        @param land_demand_df:  land demands from all techno in inputs
        @type land_demand_df: dataframe

        '''

        number_of_data = (self.year_end - self.year_start + 1)
        self.land_demand_df = land_demand_df
        self.livestock_usage_factor_df = livestock_usage_factor_df
        # set index for population df
        population_df.index = population_df['years'].values

        # Initialize demand objective  dataframe
        self.land_demand_constraint_df = pd.DataFrame(
            {'years': self.land_demand_df['years']})

        total_agriculture_surfaces = self.__extract_and_convert_superficie(
            'Habitable', 'Agriculture') / OrderOfMagnitude.magnitude_factor[OrderOfMagnitude.GIGA]

        self.land_surface_df['Agriculture total (Gha)'] = [
            total_agriculture_surfaces] * number_of_data

        self.land_surface_df.index = self.land_demand_df['years']

        # compute land use by crop and livestock

        food_crop_land_use = self.compute_food_land_use(
            population_df, self.crop_land_use_per_capita)
        food_livestock_land_use = self.compute_food_land_use(
            population_df, self.livestock_land_use_per_capita)
        food_livestock_land_use['food_land_use'] = food_livestock_land_use['food_land_use'] * \
            self.livestock_usage_factor_df['percentage'] / 100

        # remove land use by livestock and crop from available land
        self.land_surface_df['Agriculture (Gha)'] = self.land_surface_df['Agriculture total (Gha)'] - \
            food_crop_land_use['food_land_use'] - \
            food_livestock_land_use['food_land_use']

        self.land_surface_df['Crop Usage (Gha)'] = food_crop_land_use['food_land_use']
        self.land_surface_df['Livestock Usage (Gha)'] = food_livestock_land_use['food_land_use']

        self.land_surface_for_food_df = pd.DataFrame({'years': self.land_demand_df['years'],
                                                      'Agriculture total (Gha)': food_crop_land_use['food_land_use'].values +
                                                      food_livestock_land_use['food_land_use'].values})

        forest_surfaces = self.__extract_and_convert_superficie(
            'Habitable', 'Forest') / OrderOfMagnitude.magnitude_factor[OrderOfMagnitude.GIGA]
        self.land_surface_df['Forest (Gha)'] = [
            forest_surfaces] * number_of_data

        demand_crops = self.__extract_and_make_sum(
            LandUse.AGRICULTURE_TECHNO)
        demand_forest = self.__extract_and_make_sum(LandUse.FOREST_TECHNO)

        # Calculate delta for objective
        # (Convert value to million Ha)
        self.land_demand_constraint_df[self.LAND_DEMAND_CONSTRAINT_AGRICULTURE] = (self.land_surface_df['Agriculture (Gha)'].values -
                                                                                   demand_crops) / self.ref_land_use_constraint
        self.land_demand_constraint_df[self.LAND_DEMAND_CONSTRAINT_FOREST] = (forest_surfaces -
                                                                              demand_forest) / self.ref_land_use_constraint

    def get_derivative(self, objective_column, demand_column):
        """ Compute derivative of land demand objective regarding land demand

        @param objective_column: columns name to take into account in output dataframe
        @type objective_column: str

        @param demand_column:  demand_column, column to take into account in input dataframe

        @return:gradient of each constraint by demand
        """

        number_of_values = len(self.land_demand_df['years'].values)
        result = None

        if demand_column not in self.AGRICULTURE_TECHNO and demand_column not in self.FOREST_TECHNO:
            # demand_column does impact result so return an empty matrix
            result = np.identity(number_of_values) * 0.0
        else:

            if objective_column == self.LAND_DEMAND_CONSTRAINT_AGRICULTURE and demand_column in self.AGRICULTURE_TECHNO:
                result = np.identity(
                    number_of_values) * -1.0 / self.ref_land_use_constraint
            elif objective_column == self.LAND_DEMAND_CONSTRAINT_FOREST and demand_column in self.FOREST_TECHNO:
                result = np.identity(
                    number_of_values) * -1.0 / self.ref_land_use_constraint
            else:
                result = np.identity(number_of_values) * 0.0

        return result

    def d_land_demand_constraint_d_population(self, objective_column):
        """ Compute derivative of land demand objective regarding population

        @param objective_column: columns name to take into account in output dataframe
        @type objective_column: str
        @return: gradient of land demand constraint by population
        """
        number_of_values = len(self.land_demand_df['years'].values)
        result = None

        if objective_column == self.LAND_DEMAND_CONSTRAINT_AGRICULTURE:
            result = (-np.identity(
                number_of_values) * self.crop_land_use_per_capita / 1e3 - np.identity(
                number_of_values) * self.livestock_land_use_per_capita / 1e3 * self.livestock_usage_factor_df['percentage'].values / 100)
        else:
            result = np.identity(number_of_values) * 0.0

        return result

    def d_agriculture_surface_d_population(self, objective_column):
        """
        Compute derivate of land demand objectif for crop, regarding population input
        @param objective_column: columns name to take into account in output dataframe
        @type objective_column: str
        @return:gradient of surface by popultation
        """
        number_of_values = len(self.land_demand_df['years'].values)
        result = None

        if objective_column == 'Agriculture (Gha)':
            result = -np.identity(
                number_of_values) * self.crop_land_use_per_capita / 1e3 - np.identity(
                number_of_values) * self.livestock_land_use_per_capita / 1e3 * self.livestock_usage_factor_df['percentage'].values / 100
        else:
            result = np.identity(number_of_values) * 0.0

        return result

    def d_land_surface_for_food_d_population(self):
        """
        Compute derivate of land demand objectif for crop, regarding population input
        @retrun: gradient of surface by population
        """
        number_of_values = len(self.land_demand_df['years'].values)
        d_surface_d_population = np.identity(number_of_values) * (self.crop_land_use_per_capita / 1e3 +
                                                                  self.livestock_land_use_per_capita / 1e3 *
                                                                  self.livestock_usage_factor_df['percentage'].values / 100)

        return d_surface_d_population

    def d_land_surface_for_food_d_livestock_usage_factor(self, population_df):
        """
        Compute derivate of land demand objectif for crop, regarding livestock_usage_factor input
        @retrun: gradient of surface by population
        """
        number_of_values = len(self.land_demand_df['years'].values)
        d_surface_d_livestock_factor = np.identity(number_of_values) * (population_df['population'].values *
                                                                        self.livestock_land_use_per_capita / 1e3) / 100

        return d_surface_d_livestock_factor

    def __extract_and_convert_superficie(self, category, name):
        '''
        Regarding the available surface dataframe extract a specific surface value and convert into
        our unit model (ha)

        @param category: land category regarding the data
        @type category: str

        @param name: land name regarding the data
        @type name: str

        @return: number in ha unit
        '''
        surface = self.surface_df[(self.surface_df['Category'] == category) &
                                  (self.surface_df['Name'] == name)]['Surface'].values[0]
        unit = self.surface_df[(self.surface_df['Category'] == category) &
                               (self.surface_df['Name'] == name)]['Unit'].values[0]
        magnitude = self.surface_df[(self.surface_df['Category'] == category) &
                                    (self.surface_df['Name'] == name)]['Magnitude'].values[0]

        # unit conversion factor
        unit_factor = 1.0
        if unit == LandUse.KM_2_unit:
            unit_factor = self.km2toha

        magnitude_factor = 1.0
        if magnitude in OrderOfMagnitude.magnitude_factor.keys():
            magnitude_factor = OrderOfMagnitude.magnitude_factor[magnitude]

        return surface * unit_factor * magnitude_factor

    def __extract_and_make_sum(self, target_columns):
        '''
        Select columns in dataframe and make the sum of values using checks

        @param target_columns: list of columns that be taken into account
        @type target_columns: list of string

        @return: float
        '''
        dataframe_columns = list(self.land_demand_df)

        existing_columns = []
        for column in target_columns:
            if column in dataframe_columns:
                existing_columns.append(column)

        result = 0.0
        if len(existing_columns) > 0:
            result = self.land_demand_df[existing_columns].sum(
                axis=1).values

        return result

    def compute_food_land_use(self, population_df, ha_per_capita):
        '''
        compute the food land use needed for the considered population, for each year
        result = population_df * ha_per_capita

        @param population_df: df containing the population for each year
        @type population_df: dataframe
        @unit population_df: millions of people

        @params ha_per_capita: average hectares needed for food land use for one person
        @type ha_per_capita: float
        @unit ha_per_capita: ha/person

        @return: dataframe, surface used
        @unit return: billion of hectare
        '''
        land_demand_df = self.land_demand_df
        # result is in billion of ha --> need /1e3
        result = pd.DataFrame()

        result['food_land_use'] = population_df['population'] * \
            ha_per_capita / 1e3
        result.index = land_demand_df['years']

        return(result)
