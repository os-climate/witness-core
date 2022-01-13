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


class LandUseV1():
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

    TOTAL_FOOD_LAND_SURFACE = "total_food_land_surface"

    LAND_DEMAND_CONSTRAINT_DF = 'land_demand_constraint_df'
    LAND_DEMAND_CONSTRAINT_AGRICULTURE = 'Agriculture demand constraint (Gha)'
    LAND_DEMAND_CONSTRAINT_FOREST = 'Forest demand constraint (Gha)'

    LAND_SURFACE_DF = 'land_surface_df'
    LAND_SURFACE_DETAIL_DF = 'land_surface_detail_df'
    LAND_SURFACE_FOR_FOOD_DF = 'land_surface_for_food_df'

    AGRICULTURE_COLUMN = 'Agriculture (Gha)'
    FOREST_COLUMN = 'Forest (Gha)'

    # Technologies filtered by land type
    FOREST_TECHNO = ['ManagedWood (Gha)', 'UnmanagedWood (Gha)']
    AGRICULTURE_TECHNO = ['CropEnergy (Gha)', 'SolarPv (Gha)', 'SolarThermal (Gha)']

    #technologies that impact land surface constraints and coefficients
    AGRICULTURE_CONSTRAINT_IMPACT = {'Reforestation (Gha)': -1}
    FOREST_CONSTRAINT_IMPACT = {'Reforestation (Gha)': 1}


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
        self.year_start = self.param[LandUseV1.YEAR_START]
        self.year_end = self.param[LandUseV1.YEAR_END]

    def import_world_surface_data(self):
        curr_dir = os.path.dirname(__file__)
        data_file = os.path.join(curr_dir, self.surface_file)
        self.surface_df = pd.read_csv(data_file)

    def compute(self, total_food_land_surface, land_demand_df):
        ''' 
        Computation methods, comput land demands and constraints
        
        @param population_df: population from input
        @type population_df: dataframe

        @param land_demand_df:  land demands from all techno in inputs
        @type land_demand_df: dataframe
        
        '''

        number_of_data = (self.year_end - self.year_start + 1)
        self.land_demand_df = land_demand_df

        # Initialize demand objective  dataframe
        self.land_demand_constraint_df = pd.DataFrame(
            {'years': self.land_demand_df['years']})

        total_agriculture_surfaces = self.__extract_and_convert_superficie(
            'Habitable', 'Agriculture') / OrderOfMagnitude.magnitude_factor[OrderOfMagnitude.GIGA]

        # compute how much of agriculture changes because of techn
        self.land_surface_df['Added Agriculture (Gha)'] = self.__extract_and_compute_constraint_change(
            self.AGRICULTURE_CONSTRAINT_IMPACT)

        self.land_surface_df['Agriculture total (Gha)'] = [
            total_agriculture_surfaces] * number_of_data
        self.land_surface_df['Agriculture total (Gha)'] += self.land_surface_df['Added Agriculture (Gha)']

        self.land_surface_df.index = self.land_demand_df['years']


        # remove land use by food from available land
        self.land_surface_df['Agriculture (Gha)'] = self.land_surface_df['Agriculture total (Gha)'] - \
            total_food_land_surface['total surface (Gha)']

        self.land_surface_df['Food Usage (Gha)'] = total_food_land_surface['total surface (Gha)']

        self.land_surface_for_food_df = pd.DataFrame({'years': self.land_demand_df['years'],
                                                      'Agriculture total (Gha)': total_food_land_surface['total surface (Gha)'].values})

        forest_surfaces = self.__extract_and_convert_superficie(
            'Habitable', 'Forest') / OrderOfMagnitude.magnitude_factor[OrderOfMagnitude.GIGA]
        self.land_surface_df['Added Forest (Gha)'] = self.__extract_and_compute_constraint_change(self.FOREST_CONSTRAINT_IMPACT)

        self.land_surface_df['Forest (Gha)'] = [
            forest_surfaces] * number_of_data
        self.land_surface_df['Forest (Gha)'] += self.land_surface_df['Added Forest (Gha)']

        demand_crops = self.__extract_and_make_sum(
            LandUseV1.AGRICULTURE_TECHNO)
        demand_forest = self.__extract_and_make_sum(LandUseV1.FOREST_TECHNO)

        # Calculate delta for objective
        # (Convert value to million Ha)

        self.land_demand_constraint_df[self.LAND_DEMAND_CONSTRAINT_AGRICULTURE] = (self.land_surface_df['Agriculture (Gha)'].values -
                                                                                   demand_crops)
        self.land_demand_constraint_df[self.LAND_DEMAND_CONSTRAINT_FOREST] = (self.land_surface_df['Forest (Gha)'].values -
                                                                              demand_forest)

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
                    number_of_values) * -1.0
            elif objective_column == self.LAND_DEMAND_CONSTRAINT_FOREST and demand_column in self.FOREST_TECHNO:
                result = np.identity(
                    number_of_values) * -1.0
            else:
                result = np.identity(number_of_values) * 0.0

        if objective_column == self.LAND_DEMAND_CONSTRAINT_AGRICULTURE :
            result += self.d_constraint_d_surface(self.AGRICULTURE_COLUMN, demand_column)
        elif objective_column == self.LAND_DEMAND_CONSTRAINT_FOREST:
            result += self.d_constraint_d_surface(self.FOREST_COLUMN, demand_column)

        return result

    def d_constraint_d_surface(self, surface_column, demand_column):
        '''
        Compute the derivative of techno on surface constraints
        :param:surface_column, name of the key of the CONSTRAINT_TECHNO_IMPACT dict
        :type:string
        :param:demand_column, name of the land_demand_df column
        :type:string
        '''
        number_of_values = len(self.land_demand_df['years'].values)
        result = np.identity(number_of_values) * 0.0

        # get the right dictionary
        target_constraints = None
        if surface_column == self.AGRICULTURE_COLUMN:
            target_constraints = self.AGRICULTURE_CONSTRAINT_IMPACT
        elif surface_column == self.FOREST_COLUMN:
            target_constraints = self.FOREST_CONSTRAINT_IMPACT

        if target_constraints is not None and demand_column in target_constraints.keys():
            desc_matrix = np.tri(number_of_values)
            result = desc_matrix * target_constraints[demand_column]

        return result

    def d_land_demand_constraint_d_food_land_surface(self, objective_column):
        """ Compute derivative of land demand objective regarding food land surface

        @param objective_column: columns name to take into account in output dataframe
        @type objective_column: str
        @return: gradient of land demand constraint by food land surface
        """
        number_of_values = len(self.land_demand_df['years'].values)
        d_land_demand_constraint_d_food_land_surface = None

        if objective_column == self.LAND_DEMAND_CONSTRAINT_AGRICULTURE:
            d_land_demand_constraint_d_food_land_surface = -np.identity(
                number_of_values) * 1.0
        else:
            d_land_demand_constraint_d_food_land_surface = np.identity(number_of_values) * 0.0

        return d_land_demand_constraint_d_food_land_surface

    def d_agriculture_surface_d_food_land_surface(self, objective_column):
        """
        Compute derivate of land demand objectif for crop, regarding food land surface input
        @param objective_column: columns name to take into account in output dataframe
        @type objective_column: str
        @return:gradient of surface by food land surface
        """
        number_of_values = len(self.land_demand_df['years'].values)
        d_agriculture_surface_d_food_land_surface = None

        if objective_column == 'Agriculture (Gha)':
            d_agriculture_surface_d_food_land_surface = -np.identity(
                number_of_values) * 1.0
        else:
            d_agriculture_surface_d_food_land_surface = np.identity(number_of_values) * 0.0

        return d_agriculture_surface_d_food_land_surface

    def d_land_surface_for_food_d_food_land_surface(self):
        """
        Compute derivate of land demand objectif for crop, regarding food land surface input
        @retrun: gradient of surface by food land surface
        """
        number_of_values = len(self.land_demand_df['years'].values)
        d_surface_d_food_land_surface = np.identity(number_of_values) * 1.0

        return d_surface_d_food_land_surface

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
        if unit == LandUseV1.KM_2_unit:
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

    def __extract_and_compute_constraint_change(self, target_constraints):
        '''
        Select columns in the right constraint impact and compute the sum of coeff * surface

        @param target_constraint: surface key of the CONSTRAINT_TECHNO_IMPACT dict
        @type target_constraint: string

        @return: dataframe
        '''
        number_of_data = (self.year_end - self.year_start + 1)
        result = [0.0] * number_of_data
        dataframe_columns = list(self.land_demand_df)
        for surface_type in target_constraints.keys():
            if surface_type in dataframe_columns:
                coeff = [target_constraints[surface_type]] * number_of_data
                result = result + self.land_demand_df[surface_type] * coeff

        # a surface added or removed one year is also added or removed the following years
        surfaces_final = result.cumsum().values
        return surfaces_final
