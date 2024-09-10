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


class CarbonCycle():
    """
    Carbon cycle
    """
    rockstrom_limit = 450

    def __init__(self, param):
        '''
        Constructor
        '''
        self.param = param
        self.set_data()
        self.create_dataframe()

    def set_data(self):
        self.year_start = self.param[GlossaryCore.YearStart]
        self.year_end = self.param[GlossaryCore.YearEnd]
        self.time_step = self.param[GlossaryCore.TimeStep]
        self.conc_lower_strata = self.param['conc_lower_strata']
        self.conc_upper_strata = self.param['conc_upper_strata']
        self.conc_atmo = self.param['conc_atmo']
        self.init_conc_atmo = self.param['init_conc_atmo']
        self.init_upper_strata = self.param['init_upper_strata']
        self.init_lower_strata = self.param['init_lower_strata']
        self.b_twelve = self.param['b_twelve']
        self.b_twentythree = self.param['b_twentythree']
        self.b_eleven = 1.0 - self.b_twelve
        self.b_twentyone = self.b_twelve * self.conc_atmo / self.conc_upper_strata
        self.b_twentytwo = 1.0 - self.b_twentyone - self.b_twentythree
        self.b_thirtytwo = self.b_twentythree * \
            self.conc_upper_strata / self.conc_lower_strata
        self.b_thirtythree = 1.0 - self.b_thirtytwo
        self.lo_mat = self.param['lo_mat']
        self.lo_mu = self.param['lo_mu']
        self.lo_ml = self.param['lo_ml']

        self.alpha = self.param['alpha']
        self.beta = self.param['beta']
        self.ppm_ref = self.param['ppm_ref']
        self.ppm_obj = 0.
        # Conversion factor 1Gtc = 44/12 GT of CO2
        # Molar masses C02 (12+2*16=44) / C (12)
        self.gtco2_to_gtc = 44 / 12
        # conversion factor 1ppm= 2.13 Gtc
        self.gtc_to_ppm = 2.13
        self.scale_factor_carbon_cycle = self.param['scale_factor_atmo_conc']
        self.rockstrom_constraint_ref = self.param['rockstrom_constraint_ref']
        self.minimum_ppm_constraint_ref = self.param['minimum_ppm_constraint_ref']
        self.minimum_ppm_limit = self.param['minimum_ppm_limit']

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        years_range = np.arange(
            self.year_start, self.year_end + 1, self.time_step)
        self.years_range = years_range
        carboncycle_df = pd.DataFrame(index=years_range, columns=[GlossaryCore.Years,
                                                                  'atmo_conc', 'lower_ocean_conc', 'shallow_ocean_conc', 'ppm', 'atmo_share_since1850', 'atmo_share_sinceystart'])

        for key in carboncycle_df.keys():
            carboncycle_df[key] = 0
        carboncycle_df[GlossaryCore.Years] = self.years_range
        carboncycle_df.loc[self.year_start, 'atmo_conc'] = self.init_conc_atmo
        carboncycle_df.loc[self.year_start,
                           'lower_ocean_conc'] = self.init_lower_strata
        carboncycle_df.loc[self.year_start,
                           'shallow_ocean_conc'] = self.init_upper_strata
        self.carboncycle_df = carboncycle_df

        return carboncycle_df

    def compute_atmo_conc(self, year):
        """
        compute atmo conc for t using value at t-1 (MAT in DICE)
        """
        p_atmo_conc = self.carboncycle_df.at[year -
                                             self.time_step, 'atmo_conc']
        p_shallow_ocean_conc = self.carboncycle_df.at[year -
                                                      self.time_step, 'shallow_ocean_conc']
        p_emissions = self.CO2_emissions_df.at[year -
                                               self.time_step, 'total_emissions']
        atmo_conc = p_atmo_conc * self.b_eleven + p_shallow_ocean_conc * \
            self.b_twentyone + p_emissions * self.time_step / self.gtco2_to_gtc
        # Lower bound
        self.carboncycle_df.loc[year, 'atmo_conc'] = max(
            atmo_conc, self.lo_mat)
        return atmo_conc

    def compute_lower_ocean_conc(self, year):
        """
        Compute lower ocean conc at t using values at t-1
        """
        p_lower_ocean_conc = self.carboncycle_df.at[year -
                                                    self.time_step, 'lower_ocean_conc']
        p_shallow_ocean_conc = self.carboncycle_df.at[year -
                                                      self.time_step, 'shallow_ocean_conc']
        lower_ocean_conc = p_lower_ocean_conc * self.b_thirtythree + \
            p_shallow_ocean_conc * self.b_twentythree
        # Lower bound
        self.carboncycle_df.loc[year, 'lower_ocean_conc'] = max(
            lower_ocean_conc, self.lo_ml)
        return lower_ocean_conc

    def compute_upper_ocean_conc(self, year):
        """
        Compute upper ocean conc at t using values at t-1
        """
        p_lower_ocean_conc = self.carboncycle_df.at[year -
                                                    self.time_step, 'lower_ocean_conc']
        p_shallow_ocean_conc = self.carboncycle_df.at[year -
                                                      self.time_step, 'shallow_ocean_conc']
        p_atmo_conc = self.carboncycle_df.at[year -
                                             self.time_step, 'atmo_conc']
        shallow_ocean_conc = p_atmo_conc * self.b_twelve + p_shallow_ocean_conc * \
            self.b_twentytwo + p_lower_ocean_conc * self.b_thirtytwo
        # Lower Bound
        self.carboncycle_df.loc[year, 'shallow_ocean_conc'] = max(
            shallow_ocean_conc, self.lo_mu)

    def compute_ppm(self, year):
        """
         Compute Atmospheric concentrations parts per million at t
        """
        atmo_conc = self.carboncycle_df.at[year, 'atmo_conc']
        ppm = atmo_conc / self.gtc_to_ppm
        self.carboncycle_df.loc[year, 'ppm'] = ppm
        return ppm

    def compute_atmo_share(self, year):
        """
        Compute atmo share since 1850 and since 2010
        """
        atmo_conc = self.carboncycle_df.at[year, 'atmo_conc']
        init_atmo_conc = self.carboncycle_df.at[self.year_start, 'atmo_conc']

        self.CO2_emissions_df['cum_total_emissions'] = self.CO2_emissions_df['total_emissions'].cumsum()
        init_cum_total_emissions = self.CO2_emissions_df.at[self.year_start,
                                                            'cum_total_emissions']
        cum_total_emissions = self.CO2_emissions_df.at[year,
                                                       'cum_total_emissions']

        atmo_share1850 = ((atmo_conc - 588.0) /
                          (cum_total_emissions + .000001))
        atmo_shareystart = ((atmo_conc - init_atmo_conc) /
                            (cum_total_emissions - init_cum_total_emissions))

        self.carboncycle_df.loc[year,
                                'atmo_share_since1850'] = atmo_share1850
        self.carboncycle_df.loc[year,
                                'atmo_share_sinceystart'] = atmo_shareystart
        return atmo_share1850

    def compute_d_total_emissions(self):
        """
        Compute d_y / d_total_emissions, with y is a column of carboncycle_detail_df
        """
        time_step = self.time_step
        gtco2_to_gtc = self.gtco2_to_gtc
        lo_mat = self.lo_mat
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)

        b_eleven = 1.0 - self.b_twelve
        b_twentyone = self.b_twelve * self.conc_atmo / self.conc_upper_strata
        b_twentytwo = 1.0 - b_twentyone - self.b_twentythree
        b_thirtytwo = self.b_twentythree * self.conc_upper_strata / self.conc_lower_strata
        b_thirtythree = 1.0 - b_thirtytwo

        d_atmoconc_d_totalemissions = np.zeros((len(years), len(years)))
        d_swallow_d_totalemissions = np.zeros((len(years), len(years)))
        d_lower_d_totalemissions = np.zeros((len(years), len(years)))

        atmo_conc = self.carboncycle_df['atmo_conc'] / \
            self.scale_factor_carbon_cycle
        # ---- initialisation
        if atmo_conc.values[1] > lo_mat:
            d_atmoconc_d_totalemissions[1, 0] = time_step / gtco2_to_gtc

        if atmo_conc.values[2] > lo_mat:
            d_atmoconc_d_totalemissions[2, 1] = time_step / gtco2_to_gtc

        if self.carboncycle_df['shallow_ocean_conc'].values[2] > self.lo_mu:
            d_swallow_d_totalemissions[2,
                                       0] = time_step / gtco2_to_gtc * self.b_twelve
            d_atmoconc_d_totalemissions[2,
                                        0] = time_step / gtco2_to_gtc * b_eleven

        for i in range(3, len(years)):
            for j in range(0, i):
                # if lower ocean conc is not a constant, grad is not null
                if np.real(self.carboncycle_df['lower_ocean_conc'].values[i]) > self.lo_ml:

                    d_lower_d_totalemissions[i, j] = d_lower_d_totalemissions[i - 1, j] * b_thirtythree + \
                        d_swallow_d_totalemissions[i -
                                                   1, j] * self.b_twentythree

                # if shallow_ocean_conc is not a constant, grad is not null
                if np.real(self.carboncycle_df['shallow_ocean_conc'].values[i]) > self.lo_mu:
                    d_swallow_d_totalemissions[i, j] = d_atmoconc_d_totalemissions[i - 1, j] * self.b_twelve + \
                        d_swallow_d_totalemissions[i - 1, j] * b_twentytwo + \
                        d_lower_d_totalemissions[i - 1, j] * b_thirtytwo

                # if atmo_conc is not a constant, grad is not null
                if np.real(atmo_conc.values[i]) > lo_mat:
                    if j == i - 1:
                        d_atmoconc_d_totalemissions[i, j] = d_atmoconc_d_totalemissions[i - 1, j] * b_eleven + \
                            d_swallow_d_totalemissions[i - 1, j] * \
                            b_twentyone + time_step / gtco2_to_gtc
                    else:
                        d_atmoconc_d_totalemissions[i, j] = d_atmoconc_d_totalemissions[i - 1, j] * b_eleven + \
                            d_swallow_d_totalemissions[i - 1, j] * b_twentyone

            if np.real(atmo_conc.values[i]) == lo_mat:
                d_atmoconc_d_totalemissions[i, 0] = 0

        # -----------
        cum_total_emissions = self.CO2_emissions_df['cum_total_emissions']
        d_atmo1850_dtotalemission = np.zeros((len(years), len(years)))

        for i in range(0, len(years)):
            for j in range(0, len(years)):

                Cte = (
                    cum_total_emissions[self.year_start + time_step * j] + .000001)
                d_atmo1850_dtotalemission[j,
                                          i] = d_atmoconc_d_totalemissions[j, i] / Cte

        # -----------
        init_cum_total_emissions = self.CO2_emissions_df.at[self.year_start,
                                                            'cum_total_emissions']

        d_atmotoday_dtotalemission = np.zeros((len(years), len(years)))
        for i in range(0, len(years)):
            for j in range(1, len(years)):

                Cte = (cum_total_emissions[self.year_start +
                                           time_step * j] - init_cum_total_emissions)
                d_atmotoday_dtotalemission[j,
                                           i] = d_atmoconc_d_totalemissions[j, i] / Cte

                d_atmotoday_dtotalemission[0, j] = 0

        return d_atmoconc_d_totalemissions * self.scale_factor_carbon_cycle, d_lower_d_totalemissions, d_swallow_d_totalemissions, d_atmo1850_dtotalemission, d_atmotoday_dtotalemission

    def compute_d_cum_total_emissions(self):

        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)

        init_atmo_conc = self.init_conc_atmo
        init_cum_total_emissions = self.CO2_emissions_df.at[self.year_start,
                                                            'cum_total_emissions']
        cum_total_emissions = self.CO2_emissions_df['cum_total_emissions']

        atmo_conc = self.carboncycle_df['atmo_conc']
        d_atmo1850_dcumemission = np.zeros((len(years), len(years)))

        for i in range(1, len(years)):
            d_atmo1850_dcumemission[i, i] = -(atmo_conc[years[i]] - 588.0) / (
                cum_total_emissions[years[i]] + .000001) ** 2

        # -----------
        d_atmotoday_dcumtotalemission = np.zeros((len(years), len(years)))
        for i in range(1, len(years)):
            d_atmotoday_dcumtotalemission[i, i] = -(self.carboncycle_df['atmo_conc'][self.year_start + self.time_step * i] - init_atmo_conc) / (
                cum_total_emissions[self.year_start + self.time_step * i] - init_cum_total_emissions) ** 2

            d_atmotoday_dcumtotalemission[i, 0] = -\
                d_atmotoday_dcumtotalemission[i, i]

        return d_atmo1850_dcumemission, d_atmotoday_dcumtotalemission

    def compute_d_ppm(self, d_atmoconc_d_totalemissions):
        """
        Compute d_ppm / d_total_emission
        """
#         atmo_conc = self.carboncycle_df.loc[year, 'atmo_conc']
#         ppm = atmo_conc / 2.13
# rescale correctly d_atmoconc_d_totalemissions
        d_ppm = d_atmoconc_d_totalemissions / \
            self.gtc_to_ppm / self.scale_factor_carbon_cycle
        return d_ppm

    def compute_d_objective(self, d_ppm):
        delta_years = len(self.carboncycle_df)
        d_ppm_objective = (1 - self.alpha) * (1 - self.beta) * \
            d_ppm.sum(axis=0) / (self.ppm_ref * delta_years)
        return d_ppm_objective

    def compute_objective(self):
        """
        Compute ppm objective
        """
        delta_years = len(self.carboncycle_df)
        self.ppm_obj = np.asarray([(1 - self.alpha) * (1 - self.beta) * self.carboncycle_df['ppm'].sum()
                                   / (self.ppm_ref * delta_years)])

    def compute_rockstrom_limit_constraint(self):
        """
        Compute Rockstrom limit constraint
        """
        self.rockstrom_limit_constraint = -(self.carboncycle_df['ppm'].values -
                                             1.1 * self.rockstrom_limit) / self.rockstrom_constraint_ref

    def compute_minimum_ppm_limit_constraint(self):
        """
        Compute minimum ppm limit constraint
        """
        self.minimum_ppm_constraint = -\
            (self.minimum_ppm_limit -
             self.carboncycle_df['ppm'].values) / self.minimum_ppm_constraint_ref

    def compute(self, inputs_models):
        """
        Compute results of the pyworld3
        """
        self.create_dataframe()
        self.inputs_models = inputs_models
        self.CO2_emissions_df = deepcopy(self.inputs_models[GlossaryCore.CO2EmissionsDfValue])
        self.CO2_emissions_df.index = self.CO2_emissions_df[GlossaryCore.Years].values
        self.compute_ppm(self.year_start)

        for year in self.years_range[1:]:
            self.compute_atmo_conc(year)
            self.compute_lower_ocean_conc(year)
            self.compute_upper_ocean_conc(year)
            self.compute_ppm(year)
            self.compute_atmo_share(year)
        self.carboncycle_df = self.carboncycle_df.replace(
            [np.inf, -np.inf], np.nan)
        self.compute_objective()
        self.compute_rockstrom_limit_constraint()
        self.compute_minimum_ppm_limit_constraint()
        # Rescale atmo_conc of carbon_cycle_df
        self.carboncycle_df['atmo_conc'] = self.carboncycle_df['atmo_conc'] * \
            self.scale_factor_carbon_cycle

        return self.carboncycle_df.fillna(0.0), self.ppm_obj
