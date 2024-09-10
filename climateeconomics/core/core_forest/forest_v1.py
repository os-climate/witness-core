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

import numpy as np
import pandas as pd

from climateeconomics.glossarycore import GlossaryCore


class Forest():
    """
    Forest pyworld3 class
    basic for now, to evolve

    """
    YEAR_START = GlossaryCore.YearStart
    YEAR_END = GlossaryCore.YearEnd
    TIME_STEP = GlossaryCore.TimeStep
    LIMIT_DEFORESTATION_SURFACE = 'limit_deforestation_surface'
    DEFORESTATION_SURFACE = 'deforestation_surface'
    CO2_PER_HA = 'CO2_per_ha'
    INITIAL_CO2_EMISSIONS = 'initial_emissions'
    REFORESTATION_INVESTMENT = 'forest_investment'
    REFORESTATION_COST_PER_HA = 'reforestation_cost_per_ha'

    FOREST_SURFACE_DF = 'forest_surface_df'
    FOREST_DETAIL_SURFACE_DF = 'forest_surface_detail_df'
    CO2_EMITTED_FOREST_DF = GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)
    CO2_EMITTED_DETAIL_DF = GlossaryCore.CO2EmissionsDetailDfValue

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
        # forest data
        self.forest_investment = self.param[self.REFORESTATION_INVESTMENT]
        self.cost_per_ha = self.param[self.REFORESTATION_COST_PER_HA]

    def create_dataframe(self):
        """
        Create the dataframe and fill it with values at year_start
        """
        years = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years = years
        self.forest_surface_df = pd.DataFrame()
        self.CO2_emitted_df = pd.DataFrame()

    def compute(self, in_dict):
        """
        Computation methods
        """
        self.deforestation_surface = in_dict[self.DEFORESTATION_SURFACE]
        self.year_start = in_dict[self.YEAR_START]
        self.year_end = in_dict[self.YEAR_END]
        self.time_step = in_dict[self.TIME_STEP]
        self.forest_investment = in_dict[self.REFORESTATION_INVESTMENT]
        self.cost_per_ha = in_dict[self.REFORESTATION_COST_PER_HA]
        self.initial_emissions = self.param[self.INITIAL_CO2_EMISSIONS]
        years = np.arange(self.year_start, self.year_end + 1, self.time_step)
        self.limit_deforestation_surface = self.param[self.LIMIT_DEFORESTATION_SURFACE]

        self.forest_surface_df[GlossaryCore.Years] = years
        # forest surface is in Gha, deforestation_surface is in Mha,
        # deforested_surface is in Gha
        self.forest_surface_df['deforested_surface'] = - \
            self.deforestation_surface['deforested_surface'].values / 1000

        # forested surface
        # invest in G$, coest_per_ha in $/ha --> Gha
        self.forest_surface_df['forested_surface'] = self.forest_investment['forest_investment'].values / self.cost_per_ha

        # total
        self.forest_surface_df['forest_surface_evol'] = self.forest_surface_df['forested_surface'] + \
            self.forest_surface_df['deforested_surface']

        # cumulative values
        self.forest_surface_df['forest_surface_evol_cumulative'] = np.cumsum(
            self.forest_surface_df['forest_surface_evol'])
        self.forest_surface_df['deforested_surface_cumulative'] = np.cumsum(
            self.forest_surface_df['deforested_surface'])
        self.forest_surface_df['forested_surface_cumulative'] = np.cumsum(
            self.forest_surface_df['forested_surface'])

        # check limit of deforestation
        for element in range(0, len(years)):
            if self.forest_surface_df.loc[element, 'forest_surface_evol_cumulative'] < -self.limit_deforestation_surface / 1000:
                self.forest_surface_df.loc[element,
                                           'forest_surface_evol'] = 0
                self.forest_surface_df.loc[element, 'deforested_surface'] = - \
                    self.forest_surface_df.loc[element, 'forested_surface']
                self.forest_surface_df.loc[element,
                                           'forest_surface_evol_cumulative'] = -self.limit_deforestation_surface / 1000
                self.forest_surface_df.loc[element,
                                           'deforested_surface_cumulative'] = -self.forest_surface_df.loc[element, 'forested_surface_cumulative'] - self.limit_deforestation_surface / 1000

        self.CO2_emitted_df[GlossaryCore.Years] = self.years
        # in Gt of CO2
        self.CO2_emitted_df['emitted_CO2_evol'] = -self.forest_surface_df['forest_surface_evol'] * \
            self.CO2_per_ha / 1000
        self.CO2_emitted_df['emitted_CO2'] = -self.forest_surface_df['deforested_surface'] * \
            self.CO2_per_ha / 1000
        self.CO2_emitted_df['captured_CO2'] = -self.forest_surface_df['forested_surface'] * \
            self.CO2_per_ha / 1000

        self.CO2_emitted_df['emitted_CO2_evol_cumulative'] = -self.forest_surface_df['forest_surface_evol_cumulative'] * \
            self.CO2_per_ha / 1000 + self.initial_emissions
        self.CO2_emitted_df['emitted_CO2_cumulative'] = -self.forest_surface_df['deforested_surface_cumulative'] * \
            self.CO2_per_ha / 1000 + self.initial_emissions
        self.CO2_emitted_df['captured_CO2_cumulative'] = -self.forest_surface_df['forested_surface_cumulative'] * \
            self.CO2_per_ha / 1000
        # To make forest_disc work with forest_model v1 & v2
        self.forest_surface_df['global_forest_surface'] = np.zeros(len(years))

    # Gradients
    def d_deforestation_surface_d_deforestation_surface(self, ):
        """
        Compute gradient of deforestation surface by deforestation_surface (design variable)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_deforestation_surface_d_forests = np.identity(number_of_values)
        for i in range(0, number_of_values):
            if self.forest_surface_df.loc[i, 'forest_surface_evol_cumulative'] != -self.limit_deforestation_surface / 1000:
                d_deforestation_surface_d_forests[i][i] = - 1 / 1000
            else:
                d_deforestation_surface_d_forests[i][i] = 0

        return d_deforestation_surface_d_forests

    def d_forestation_surface_d_invest(self, ):
        """
        Compute gradient of deforestation surface by deforestation_surface (design variable)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_forestation_surface_d_invest = np.identity(number_of_values)
        for i in range(0, number_of_values):
            if self.forest_surface_df.loc[i, 'forest_surface_evol_cumulative'] != -self.limit_deforestation_surface / 1000:
                d_forestation_surface_d_invest[i][i] = 1 / self.cost_per_ha
            else:
                d_forestation_surface_d_invest[i][i] = 0

        return d_forestation_surface_d_invest

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

        d_CO2_emitted = - d_deforestation_surface * self.CO2_per_ha / 1000

        return d_CO2_emitted
