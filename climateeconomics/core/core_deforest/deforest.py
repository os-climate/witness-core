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


class Deforest():
    """
    Deforestation model class 

    basic for now, to evolve 

    """

    YEAR_START = 'year_start'
    YEAR_END = 'year_end'
    TIME_STEP = 'time_step'
    FOREST_DF = 'forest_df'
    DEFORESTATION_RATE_DF = 'forest_evolution_rate_df'
    CO2_PER_HA = 'CO2_per_ha'

    DEFORESTED_SURFACE_DF = 'forest_surface_evol_df'
    NON_CAPTURED_CO2_DF = 'captured_CO2_evol_df'

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
        # surface of the world forests
        self.forest_df = self.param[self.FOREST_DF]
        # percentage of deforestation
        self.deforestation_rate_df = self.param[self.DEFORESTATION_RATE_DF]
        # kg of CO2 not absorbed for 1 ha of forest deforested
        self.CO2_per_ha = self.param[self.CO2_PER_HA]

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
        self.non_captured_CO2_df = pd.DataFrame()

    def compute(self, in_dict):
        """
        Computation methods
        """
        self.forest_df = in_dict[self.FOREST_DF]
        self.deforestation_rate_df = in_dict[self.DEFORESTATION_RATE_DF]

        self.deforested_surface_df['years'] = self.years
        # forest surface is in Gha, deforestation_rate is in %,
        # deforested_surface is in Gha
        self.deforested_surface_df['forest_surface_evol'] = -self.forest_df['forest_surface'] * \
            self.deforestation_rate_df['forest_evolution'] / 100.0

        self.deforested_surface_df['forest_surface_evol_cumulative'] = np.cumsum(
            self.deforested_surface_df['forest_surface_evol'])

        self.non_captured_CO2_df['years'] = self.years
        # in Mt of CO2
        self.non_captured_CO2_df['captured_CO2_evol'] = self.deforested_surface_df['forest_surface_evol'] * \
            self.CO2_per_ha
        self.non_captured_CO2_df['captured_CO2_evol_cumulative'] = self.deforested_surface_df['forest_surface_evol_cumulative'] * \
            self.CO2_per_ha

    def d_deforestation_surface_d_forests(self, ):
        """
        Compute gradient of deforestation surface by forests surface (inupt from land use)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        d_deforestation_surface_d_forests = idty * \
            self.deforestation_rate_df['forest_evolution'].values / 100.0

        return d_deforestation_surface_d_forests

    def d_cum(self, derivative):
        """
        compute the gradient of a cumulative derivative
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_cum = np.identity(number_of_values)
        for i in range(0, number_of_values):
            d_cum[i] = derivative[i]
            if i > 0:
                d_cum[i] += d_cum[i - 1]
        return d_cum

    def d_deforestation_surface_d_deforestation_rate(self):
        """
        Compute gradient of deforestation surface by deforestation rate
        """
        number_of_values = (self.year_end - self.year_start + 1)
        idty = np.identity(number_of_values)
        d_deforestation_surface_d_deforestation_rate = idty * \
            self.forest_df['forest_surface'].values / 100.0

        return d_deforestation_surface_d_deforestation_rate

    def d_non_captured_CO2(self, d_deforestation_surface):
        """
        Compute gradient of non_captured_CO2 by deforestation surface
        :param: d_deforestation_surface, derivative of deforestation surface
        """
        d_non_captured_CO2 = d_deforestation_surface * self.CO2_per_ha

        return d_non_captured_CO2
