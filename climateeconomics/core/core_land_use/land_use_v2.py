'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/15-2023/11/03 Copyright 2023 Capgemini

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

import os

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


class LandUseV2():
    """
    Land use pyworld3 class

    basic for now, to evolve 

    source: https://ourworldindata.org/land-use
    """

    KM_2_unit = 'km2'
    HECTARE = 'ha'

    LAND_DEMAND_DF = 'land_demand_df'
    YEAR_START = GlossaryCore.YearStart
    YEAR_END = GlossaryCore.YearEnd
    INIT_UNMANAGED_FOREST_SURFACE = 'initial_unmanaged_forest_surface'

    TOTAL_FOOD_LAND_SURFACE = 'total_food_land_surface'
    FOREST_SURFACE_DF = 'forest_surface_df'

    LAND_DEMAND_CONSTRAINT = 'land_demand_constraint'
    LAND_DEMAND_CONSTRAINT_REF = 'land_demand_constraint_ref'

    LAND_SURFACE_DF = 'land_surface_df'
    LAND_SURFACE_DETAIL_DF = 'land_surface_detail_df'
    LAND_SURFACE_FOR_FOOD_DF = 'land_surface_for_food_df'

    AGRICULTURE_COLUMN = 'Agriculture (Gha)'
    FOREST_COLUMN = 'Forest (Gha)'

    # Technologies filtered by land type
    FOREST_TECHNO = ['']
    AGRICULTURE_TECHNO = ['Crop (Gha)', 'SolarPv (Gha)', 'SolarThermal (Gha)']


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
        self.land_demand_constraint = None
        self.land_surface_df = pd.DataFrame()

    def set_data(self):
        self.year_start = self.param[LandUseV2.YEAR_START]
        self.year_end = self.param[LandUseV2.YEAR_END]
        self.nb_years = self.year_end - self.year_start + 1
        self.ref_land_use_constraint = self.param[LandUseV2.LAND_DEMAND_CONSTRAINT_REF]

    def import_world_surface_data(self):
        curr_dir = os.path.dirname(__file__)
        data_file = os.path.join(curr_dir, self.surface_file)
        self.surface_df = pd.read_csv(data_file)
        self.total_agriculture_surfaces = self.__extract_and_convert_superficie('Habitable', GlossaryCore.SectorAgriculture) / \
                                          OrderOfMagnitude.magnitude_factor[OrderOfMagnitude.GIGA]
        self.total_forest_surfaces = self.__extract_and_convert_superficie('Habitable', 'Forest') / \
                                     OrderOfMagnitude.magnitude_factor[OrderOfMagnitude.GIGA]
        self.total_shrub_surfaces = self.__extract_and_convert_superficie('Habitable', 'Shrub') / \
                                     OrderOfMagnitude.magnitude_factor[OrderOfMagnitude.GIGA]


    def compute(self, land_demand_df, total_food_land_surface, total_forest_surface_df):
        ''' 
        Computation methods, comput land demands and constraints

        @param land_demand_df:  land demands from all techno in inputs
        @type land_demand_df: dataframe

        '''

        number_of_data = (self.year_end - self.year_start + 1)
        self.land_demand_df = land_demand_df

        # add years
        self.land_surface_df[GlossaryCore.Years] = np.arange(self.year_start, self.year_end + 1, 1)
        #------------------------------------------------
        # Add available surfaces to df
        self.land_surface_df['Available Agriculture Surface (Gha)'] = np.ones(number_of_data)*self.total_agriculture_surfaces
        self.land_surface_df['Available Forest Surface (Gha)'] = np.ones(number_of_data)*self.total_forest_surfaces
        self.land_surface_df['Available Shrub Surface (Gha)'] = np.ones(number_of_data)*self.total_shrub_surfaces

        #------------------------------------------------
        # Add global forest and food surface from agriculture mix sub models
        self.land_surface_df['Forest Surface (Gha)'] = total_forest_surface_df['global_forest_surface'].values
        self.land_surface_df['Total Forest Surface (Gha)'] = self.land_surface_df['Forest Surface (Gha)']
        self.land_surface_df['Food Surface (Gha)'] = total_food_land_surface['total surface (Gha)'].values
        self.land_surface_df['Total Agriculture Surface (Gha)'] = self.land_surface_df['Food Surface (Gha)']

        # Loop on techno using agriculture or forest surfaces
        agri_techno = []
        forest_techno = []
        land_demand_columns = list(self.land_demand_df)
        for techno in LandUseV2.AGRICULTURE_TECHNO:
            if techno in land_demand_columns:
                agri_techno.append(techno)
        for techno in LandUseV2.FOREST_TECHNO:
            if techno in land_demand_columns:
                forest_techno.append(techno)
        for techno in agri_techno:
            self.land_surface_df[techno] = self.land_demand_df[techno]
            self.land_surface_df['Total Agriculture Surface (Gha)'] += self.land_surface_df[techno]
        for techno in forest_techno:
            self.land_surface_df[techno] = self.land_demand_df[techno]
            self.land_surface_df['Total Forest Surface (Gha)'] += self.land_surface_df[techno]

        # Calculate the land_use_constraint
        # By comparing the total available land surface to the demands
        self.land_demand_constraint = np.asarray((\
            (self.total_agriculture_surfaces + self.total_forest_surfaces + self.total_shrub_surfaces) -\
            (self.land_surface_df['Total Agriculture Surface (Gha)'] +
             self.land_surface_df['Total Forest Surface (Gha)']))\
                                      / self.ref_land_use_constraint)

    def __extract_and_convert_superficie(self, category, name):
        '''
        Regarding the available surface dataframe extract a specific surface value and convert into
        our unit pyworld3 (ha)

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
        if unit == LandUseV2.KM_2_unit:
            unit_factor = self.km2toha

        magnitude_factor = 1.0
        if magnitude in OrderOfMagnitude.magnitude_factor.keys():
            magnitude_factor = OrderOfMagnitude.magnitude_factor[magnitude]

        return surface * unit_factor * magnitude_factor

