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
import numpy as np
from pandas.core.frame import DataFrame

from climateeconomics.glossarycore import GlossaryCore


class TempChange(object):
    """
     Temperature evolution
    """

    def __init__(self):
        '''
        Constructor
        '''
        self.carboncycle_df = None

    def set_data(self, inputs):
        self.year_start = inputs[GlossaryCore.YearStart]
        self.year_end = inputs[GlossaryCore.YearEnd]
        self.init_temp_ocean = inputs['init_temp_ocean']
        self.init_temp_atmo = inputs['init_temp_atmo']
        self.eq_temp_impact = inputs['eq_temp_impact']
        self.init_forcing_nonco = inputs['init_forcing_nonco']
        self.hundred_forcing_nonco = inputs['hundred_forcing_nonco']
        self.climate_upper = inputs['climate_upper']
        self.transfer_upper = inputs['transfer_upper']
        self.transfer_lower = inputs['transfer_lower']
        self.forcing_eq_co2 = inputs['forcing_eq_co2']
        self.lo_tocean = inputs['lo_tocean']
        self.up_tatmo = inputs['up_tatmo']
        self.up_tocean = inputs['up_tocean']

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        years_range = np.arange(
            self.year_start,
            self.year_end + 1)
        self.years_range = years_range
        temperature_df = DataFrame(
            index=years_range,
            columns=[
                GlossaryCore.Years,
                GlossaryCore.ExoGForcing,
                GlossaryCore.Forcing,
                GlossaryCore.TempAtmo,
                GlossaryCore.TempOcean])
        temperature_df.loc[self.year_start,
                           GlossaryCore.TempOcean] = self.init_temp_ocean
        temperature_df.loc[self.year_start, GlossaryCore.TempAtmo] = self.init_temp_atmo
        temperature_df[GlossaryCore.Years] = self.years_range
        self.temperature_df = temperature_df
        return temperature_df

    def compute_exog_forcing(self, year):
        """
        Compute exogenous forcing for other greenhouse gases
        """
        t = (year - self.year_start) + 1
        exog_forcing = None  # initialize exog_forcing variable defined in either if or else statement
        if t < 18:
            exog_forcing = self.init_forcing_nonco + \
                (1. / 17.) * (self.hundred_forcing_nonco -
                              self.init_forcing_nonco) * (t - 1)
        elif t >= 18:
            exog_forcing = self.init_forcing_nonco + \
                (self.hundred_forcing_nonco - self.init_forcing_nonco)
        self.temperature_df.loc[year, GlossaryCore.ExoGForcing] = exog_forcing
        return exog_forcing

    def compute_forcing(self, year):
        """
        Compute increase in radiative forcing for t using values at t-1
        (watts per m2 from 1900)
        """
        atmo_conc = self.carboncycle_df.loc[year, 'atmo_conc']
        exog_forcing = self.temperature_df.loc[year, GlossaryCore.ExoGForcing]
        forcing = self.forcing_eq_co2 * \
            ((np.log((atmo_conc) / 588.)) / np.log(2)) + exog_forcing
        self.temperature_df.loc[year, GlossaryCore.Forcing] = forcing
        return forcing

    def compute_temp_atmo(self, year):
        """
        Compute temperature of atmosphere (t) using t-1 values

        """
        p_temp_atmo = self.temperature_df.loc[year - 1, GlossaryCore.TempAtmo]
        p_temp_ocean = self.temperature_df.loc[year - 1, GlossaryCore.TempOcean]
        forcing = self.temperature_df.loc[year, GlossaryCore.Forcing]
        temp_atmo = p_temp_atmo + self.climate_upper * \
            ((forcing - (self.forcing_eq_co2 / self.eq_temp_impact) *
              p_temp_atmo) - (self.transfer_upper * (p_temp_atmo - p_temp_ocean)))
        # Lower bound
        self.temperature_df.loc[year, GlossaryCore.TempAtmo] = min(
            temp_atmo, self.up_tatmo)
        return temp_atmo

    def compute_temp_ocean(self, year):
        """
        Compute temperature of lower ocean  at t using t-1 values
        """
        p_temp_ocean = self.temperature_df.loc[year - 1, GlossaryCore.TempOcean]
        p_temp_atmo = self.temperature_df.loc[year - 1, GlossaryCore.TempAtmo]
        temp_ocean = p_temp_ocean + self.transfer_lower * \
            (p_temp_atmo - p_temp_ocean)
        # Bounds
        temp_ocean = max(temp_ocean, self.lo_tocean)
        self.temperature_df.loc[year, GlossaryCore.TempOcean] = min(
            temp_ocean, self.up_tocean)
        return temp_ocean

    def compute(self, in_dict):
        """
        Compute all
        """
        self.carboncycle_df = in_dict.pop(GlossaryCore.CarbonCycleDfValue)

        self.set_data(in_dict)
        self.create_dataframe()
        self.carboncycle_df = self.carboncycle_df.set_index(self.years_range)
        self.compute_exog_forcing(self.year_start)
        self.compute_forcing(self.year_start)
        for year in self.years_range[1:]:
            self.compute_exog_forcing(year)
            self.compute_forcing(year)
            self.compute_temp_atmo(year)
            self.compute_temp_ocean(year)

        self.temperature_df = self.temperature_df.replace(
            [np.inf, -np.inf], np.nan)
        return self.temperature_df.fillna(0.0)
