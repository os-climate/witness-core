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


class Forest():
    """
    Forest model class 
    basic for now, to evolve 

    """
    YEAR_START = 'year_start'
    YEAR_END = 'year_end'
    TIME_STEP = 'time_step'
    LIMIT_DEFORESTATION_SURFACE = 'limit_deforestation_surface'
    DEFORESTATION_SURFACE = 'deforestation_surface'
    CO2_PER_HA = 'CO2_per_ha'
    INITIAL_CO2_EMISSIONS = 'initial_emissions'

    DEFORESTED_SURFACE_DF = 'deforested_surface_df'
    CO2_EMITTED_FOREST_DF = 'CO2_emitted_forest_df'

    def __init__(self, param):
        """
        Constructor
        """
        self.param = param
        self.set_data()
        self.create_dataframe()

    def set_data(self):
        self.year_start = self.param[self.YEAR_START]
        self.year_end = self.param[self.YEAR_END]
        self.time_step = self.param[self.TIME_STEP]
        # deforestation limite
        self.limit_deforestation_surface = self.param[self.LIMIT_DEFORESTATION_SURFACE]
        # percentage of deforestation
        self.deforestation_surface = self.param[self.DEFORESTATION_SURFACE]
        # kg of CO2 not absorbed for 1 ha of forest deforested
        self.CO2_per_ha = self.param[self.CO2_PER_HA]
        # initial CO2 emissions
        self.initial_emissions = self.param[self.INITIAL_CO2_EMISSIONS]

    def create_dataframe(self):
        """
        Create the dataframe and fill it with values at year_start
        """
        years = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years = years
        self.deforested_surface_df = pd.DataFrame()
        self.CO2_emitted_df = pd.DataFrame()

    def compute(self, in_dict):
        """
        Computation methods
        """
        self.deforestation_surface = in_dict[self.DEFORESTATION_SURFACE]
        self.year_start = in_dict[self.YEAR_START]
        self.year_end = in_dict[self.YEAR_END]
        self.time_step = in_dict[self.TIME_STEP]
        self.initial_emissions = self.param[self.INITIAL_CO2_EMISSIONS]
        years = np.arange(self.year_start, self.year_end + 1, self.time_step)
        limit_deforestation_surface = self.param[self.LIMIT_DEFORESTATION_SURFACE]

        self.deforested_surface_df['years'] = years
        # forest surface is in Gha, deforestation_surface is in Mha,
        # deforested_surface is in Gha
        self.deforested_surface_df['forest_surface_evol'] = - \
            self.deforestation_surface['deforested_surface'] / 1000

        self.deforested_surface_df['forest_surface_evol_cumulative'] = np.cumsum(
            self.deforested_surface_df['forest_surface_evol'])

        # check limit of deforestation
        for element in range(0, len(years)):
            if self.deforested_surface_df.loc[element, 'forest_surface_evol_cumulative'] < -limit_deforestation_surface / 1000:
                self.deforested_surface_df.loc[element,
                                               'forest_surface_evol'] = 0
                self.deforested_surface_df.loc[element,
                                               'forest_surface_evol_cumulative'] = -limit_deforestation_surface / 1000

        self.CO2_emitted_df['years'] = self.years
        # in Mt of CO2
        self.CO2_emitted_df['emitted_CO2_evol'] = -self.deforested_surface_df['forest_surface_evol'] * \
            self.CO2_per_ha
        self.CO2_emitted_df['emitted_CO2_evol_cumulative'] = -self.deforested_surface_df['forest_surface_evol_cumulative'] * \
            self.CO2_per_ha + self.initial_emissions

    def d_deforestation_surface_d_deforestation_surface(self, ):
        """
        Compute gradient of deforestation surface by deforestation_surface (design variable)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_deforestation_surface_d_forests = np.identity(number_of_values)
        for i in range(0, number_of_values):
            if self.deforested_surface_df.loc[i, 'forest_surface_evol'] != 0:
                d_deforestation_surface_d_forests[i][i] = -1 / 1000
            else:
                d_deforestation_surface_d_forests[i][i] = 0

        return d_deforestation_surface_d_forests

    def d_cum(self, derivative):
        """
        compute the gradient of a cumulative derivative
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_cum = np.identity(number_of_values)
        for i in range(0, number_of_values):
            d_cum[i] = derivative[i]
            if derivative[i][i] != 0:
                if i > 0:
                    d_cum[i] += d_cum[i - 1]
        return d_cum

    def d_CO2_emitted(self, d_deforestation_surface):
        """
        Compute gradient of non_captured_CO2 by deforestation surface
        :param: d_deforestation_surface, derivative of deforestation surface
        """

        d_CO2_emitted = - d_deforestation_surface * self.CO2_per_ha

        return d_CO2_emitted
