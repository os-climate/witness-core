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
from copy import deepcopy

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


class TempChange(object):
    """
     Temperature evolution
    """

    # Constant
    LAST_TEMPERATURE_OBJECTIVE = 'last_temperature'
    INTEGRAL_OBJECTIVE = 'integral'

    def __init__(self, inputs):
        '''
        Constructor
        '''
        self.carboncycle_df = None
        self.forcing_df = None
        self.temperature_objective = None
        self.ppm_to_gtc = 2.13
        self.set_data(inputs)
        self.create_dataframe()

    def set_data(self, inputs):
        self.year_start = inputs['year_start']
        self.year_end = inputs['year_end']
        self.time_step = inputs['time_step']
        self.init_temp_ocean = inputs['init_temp_ocean']
        self.init_temp_atmo = inputs['init_temp_atmo']
        self.eq_temp_impact = inputs['eq_temp_impact']

        self.forcing_model = inputs['forcing_model']

        if self.forcing_model == 'DICE':
            self.init_forcing_nonco = inputs['init_forcing_nonco']
            self.hundred_forcing_nonco = inputs['hundred_forcing_nonco']
        else:
            self.ch4_conc_init_ppm = inputs['pre_indus_ch4_concentration_ppm']
            self.n2o_conc_init_ppm = inputs['pre_indus_n2o_concentration_ppm']

        self.climate_upper = inputs['climate_upper']
        self.transfer_upper = inputs['transfer_upper']
        self.transfer_lower = inputs['transfer_lower']
        self.forcing_eq_co2 = inputs['forcing_eq_co2']
        self.c0_ppm = inputs['pre_indus_co2_concentration_ppm']
        self.lo_tocean = inputs['lo_tocean']
        self.up_tatmo = inputs['up_tatmo']
        self.up_tocean = inputs['up_tocean']

        self.alpha = inputs['alpha']
        self.beta = inputs['beta']
        self.temperature_obj_option = inputs['temperature_obj_option']

        self.scale_factor_carbon_cycle = inputs['scale_factor_atmo_conc']

        self.temperature_change_ref = inputs['temperature_change_ref']
        # rescale atmo_conc of carbon_cycle_df
        if inputs['carboncycle_df'] is not None:
            self.carboncycle_df = pd.DataFrame({'years': inputs['carboncycle_df']['years'].values,
                                                'atmo_conc': inputs['carboncycle_df']['atmo_conc'].values /
                                                self.scale_factor_carbon_cycle})

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        years_range = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years_range = years_range
        temperature_df = DataFrame(
            index=years_range,
            columns=['years',
                     'exog_forcing',
                     'forcing',
                     'temp_atmo',
                     'temp_ocean'])
        for key in temperature_df.keys():
            temperature_df[key] = 0
        temperature_df['years'] = years_range
        temperature_df.loc[self.year_start,
                           'temp_ocean'] = self.init_temp_ocean
        temperature_df.loc[self.year_start, 'temp_atmo'] = self.init_temp_atmo
        self.temperature_df = temperature_df

        self.forcing_df = DataFrame({'years': self.years_range})
        return temperature_df

    def compute_exog_forcing_dice(self):
        """
        Compute exogenous forcing for other greenhouse gases following DICE model
        linear increase of exogenous forcing following a given scenario
        """

        exog_forcing = np.linspace(
            self.init_forcing_nonco, self.hundred_forcing_nonco, len(self.years_range))

        self.forcing_df['CH4 and N20 forcing'] = exog_forcing
        return exog_forcing

    def compute_exog_forcing_myhre(self, atmo_conc):
        """
        Compute exogenous forcing for CH4 and N2O gases following Myhre model
        Myhre et al, 1998, JGR, doi: 10.1029/98GL01908
        Myhre model can be found in FUND, MAGICC and FAIR IAM models

        in FUND 0.036 * 1.4(np.sqrt(ch4_conc) - np.sqrt(ch4_conc_init))
        in FAIR 0.036 (np.sqrt(ch4_conc) -*np.sqrt(ch4_conc_init))

        we suppose that ch4 concentration and n2o cocnentration are varying the same way as co2_concentration
        We use ppm unit to compute this forcing as in FAIR and FUND
        """
        ch4_conc = atmo_conc / self.ppm_to_gtc * self.ch4_conc_init_ppm / self.c0_ppm
        n2o_conc = atmo_conc / self.ppm_to_gtc * self.n2o_conc_init_ppm / self.c0_ppm

        def MN(c1, c2):
            return 0.47 * np.log(1 + 2.01e-5 * (c1 * c2)**(0.75) +
                                 5.31e-15 * c1 * (c1 * c2)**(1.52))

        exog_forcing = 0.036 * (np.sqrt(ch4_conc) - np.sqrt(self.ch4_conc_init_ppm)) - (
            MN(ch4_conc, self.n2o_conc_init_ppm) - MN(self.ch4_conc_init_ppm, self.n2o_conc_init_ppm)) + \
            0.12 * (np.sqrt(n2o_conc) - np.sqrt(self.n2o_conc_init_ppm)) - (
            MN(self.ch4_conc_init_ppm, n2o_conc) - MN(self.ch4_conc_init_ppm, self.n2o_conc_init_ppm))

        self.forcing_df['CH4 and N20 forcing'] = exog_forcing
        return exog_forcing

    def compute_forcing_etminan(self, atmo_conc):
        """
        Compute exogenous forcing for other greenhouse gases following DICE model
        linear increase of exogenous forcing following a given scenario
        """

        ch4_conc = atmo_conc / self.ppm_to_gtc * self.ch4_conc_init_ppm / self.c0_ppm
        n2o_conc = atmo_conc / self.ppm_to_gtc * self.n2o_conc_init_ppm / self.c0_ppm
        atmo_conc_ppm = atmo_conc / self.ppm_to_gtc

        co2mean = 0.5 * (atmo_conc / self.ppm_to_gtc + self.c0_ppm)
        ch4mean = 0.5 * (ch4_conc + self.ch4_conc_init_ppm)
        n2omean = 0.5 * (n2o_conc + self.n2o_conc_init_ppm)

        co2_forcing = (-2.4e-7 * (atmo_conc_ppm - self.c0_ppm)**2 + 7.2e-4 * np.fabs(atmo_conc_ppm - self.c0_ppm) -
                       2.1e-4 * n2omean + self.forcing_eq_co2 / np.log(2)) * np.log(atmo_conc_ppm / self.c0_ppm)
        ch4_forcing = (-1.3e-6 * ch4mean - 8.2e-6 * n2omean + 0.043) * (np.sqrt(n2o_conc) -
                                                                        np.sqrt(self.n2o_conc_init_ppm))
        n2o_forcing = (-8.0e-6 * co2mean + 4.2e-6 * n2omean - 4.9e-6 * ch4mean + 0.117) * \
            (np.sqrt(n2o_conc) - np.sqrt(self.n2o_conc_init_ppm))

        self.forcing_df['CO2 forcing'] = co2_forcing
        self.forcing_df['CH4 forcing'] = ch4_forcing
        self.forcing_df['N2O forcing'] = n2o_forcing

        return co2_forcing + ch4_forcing + n2o_forcing

    def compute_forcing_meinshausen(self, atmo_conc):
        """
        Compute exogenous forcing for other greenhouse gases following DICE model
        linear increase of exogenous forcing following a given scenario
        """
        a1 = -2.4785e-07
        b1 = 0.00075906
        c1 = -0.0021492
        d1 = 5.2488
        a2 = -0.00034197
        b2 = 0.00025455
        c2 = -0.00024357
        d2 = 0.12173
        a3 = -8.9603e-05
        b3 = -0.00012462
        d3 = 0.045194

        ch4_conc = atmo_conc / self.ppm_to_gtc * self.ch4_conc_init_ppm / self.c0_ppm
        n2o_conc = atmo_conc / self.ppm_to_gtc * self.n2o_conc_init_ppm / self.c0_ppm
        atmo_conc_ppm = atmo_conc / self.ppm_to_gtc
        Camax = self.c0_ppm - b1 / (2 * a1)
        # if self.c0_ppm < atmo_conc_ppm <= Camax:  # the most likely case
        alphap = d1 + a1 * (atmo_conc_ppm - self.c0_ppm)**2 + \
            b1 * (atmo_conc_ppm - self.c0_ppm)

        alphap[atmo_conc_ppm <= self.c0_ppm] = d1
        alphap[atmo_conc_ppm >= Camax] = d1 - b1**2 / (4 * a1)

        alphaN2O = c1 * np.sqrt(n2o_conc)
        co2_forcing = (alphap + alphaN2O) * np.log(atmo_conc_ppm / self.c0_ppm)
        self.forcing_df['CO2 forcing'] = co2_forcing

        # CH4
        ch4_forcing = (
            a3 * np.sqrt(ch4_conc) + b3 * np.sqrt(n2o_conc) + d3) * (np.sqrt(ch4_conc) - np.sqrt(self.ch4_conc_init_ppm))

        self.forcing_df['CH4 forcing'] = ch4_forcing
        # N2O
        n2o_forcing = (a2 * np.sqrt(atmo_conc_ppm) + b2 * np.sqrt(n2o_conc) +
                       c2 * np.sqrt(ch4_conc) + d2) * (np.sqrt(n2o_conc) - np.sqrt(self.n2o_conc_init_ppm))

        self.forcing_df['N2O forcing'] = n2o_forcing

        return co2_forcing + ch4_forcing + n2o_forcing

    def compute_forcing(self):
        """
        Compute increase in radiative forcing for t using values at t-1
        (watts per m2 from 1900)
        """
        atmo_conc = self.carboncycle_df['atmo_conc'].values

        if self.forcing_model == 'DICE':

            exog_forcing = self.compute_exog_forcing_dice()
            co2_forcing = self.compute_log_co2_forcing(atmo_conc)
            self.forcing_df['CO2 forcing'] = co2_forcing
            forcing = co2_forcing + exog_forcing

        elif self.forcing_model == 'Myhre':

            exog_forcing = self.compute_exog_forcing_myhre(atmo_conc)
            co2_forcing = self.compute_log_co2_forcing(atmo_conc)
            self.forcing_df['CO2 forcing'] = co2_forcing
            forcing = co2_forcing + exog_forcing

        elif self.forcing_model == 'Etminan':

            forcing = self.compute_forcing_etminan(atmo_conc)

        elif self.forcing_model == 'Meinshausen':

            forcing = self.compute_forcing_meinshausen(atmo_conc)

        self.temperature_df['forcing'] = forcing

    def compute_log_co2_forcing(self, atmo_conc):

        co2_forcing = self.forcing_eq_co2 / np.log(2) * \
            np.log(atmo_conc / (self.c0_ppm * self.ppm_to_gtc))

        return co2_forcing

    def compute_temp_atmo(self, year):
        """
        Compute temperature of atmosphere (t) using t-1 values
        """
        p_temp_atmo = self.temperature_df.at[year -
                                             self.time_step, 'temp_atmo']
        p_temp_ocean = self.temperature_df.at[year -
                                              self.time_step, 'temp_ocean']
        forcing = self.temperature_df.at[year, 'forcing']
        temp_atmo = p_temp_atmo + (self.climate_upper / (5.0 / self.time_step)) * \
            ((forcing - (self.forcing_eq_co2 / self.eq_temp_impact) *
              p_temp_atmo) - ((self.transfer_upper / (5.0 / self.time_step)) * (p_temp_atmo - p_temp_ocean)))
        # Lower bound
        self.temperature_df.loc[year, 'temp_atmo'] = min(
            temp_atmo, self.up_tatmo)
        return temp_atmo

    def compute_temp_ocean(self, year):
        """
        Compute temperature of lower ocean  at t using t-1 values
        """
        p_temp_ocean = self.temperature_df.at[year -
                                              self.time_step, 'temp_ocean']
        p_temp_atmo = self.temperature_df.at[year -
                                             self.time_step, 'temp_atmo']
        temp_ocean = p_temp_ocean + (self.transfer_lower / (5.0 / self.time_step)) * \
            (p_temp_atmo - p_temp_ocean)
        # Bounds
        temp_ocean = max(temp_ocean, self.lo_tocean)
        self.temperature_df.loc[year, 'temp_ocean'] = min(
            temp_ocean, self.up_tocean)
        return temp_ocean

    ######### GRADIENTS ########

    def compute_d_forcing(self):
        # forcing = self.forcing_eq_co2 * (np.log(atmo_conc / 588.) / np.log(2) + exog_forcing
        # forcing only depends of atmo_conc
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)

        atmo_conc = self.carboncycle_df['atmo_conc'].values * \
            self.scale_factor_carbon_cycle

        d_forcing = np.identity(len(years)) * self.forcing_eq_co2 / \
            (np.log(2) * self.carboncycle_df['atmo_conc'].values)

        return d_forcing

    def compute_d_temp_atmo(self):

        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)
        atmo_conc = self.carboncycle_df['atmo_conc'].values

        # derivative matrix initialization
        d_tempatmo_d_atmoconc = np.identity(nb_years) * 0
        d_tempocean_d_atmoconc = np.identity(nb_years) * 0
        # first line stays at zero since derivatives of initial values are zero
        # second line is only equal to the derivative of forcing effect
        d_tempatmo_d_atmoconc[1, 1] = self.climate_upper * self.time_step / 5.0 \
            * self.forcing_eq_co2 / (np.log(2) * atmo_conc[1])

        for i in range(2, nb_years):
            j = 1
            # setting of the matrix diagonal which represents the forcing
            # component
            d_tempatmo_d_atmoconc[i, i] = self.climate_upper * self.time_step / 5.0 \
                * self.forcing_eq_co2 / (np.log(2) * atmo_conc[i])
            # if temp_atmo is saturated at up_tatmo, it won't depend on atmo_conc anymore
            # so the derivative will be zero
            # if temp_ocean is saturated it has no effect as it only depends on
            # temp_atmo
            if (self.temperature_df.at[years[i], 'temp_atmo'] == self.up_tatmo):
                d_tempatmo_d_atmoconc[i, i] = 0

            while j < i:
                #-------atmo temp derivative------------
                d_tempatmo_d_atmoconc[i, j] = d_tempatmo_d_atmoconc[i - 1, j] \
                    - self.climate_upper * self.time_step / 5.0 * self.forcing_eq_co2 / self.eq_temp_impact * d_tempatmo_d_atmoconc[i - 1, j] \
                    - self.climate_upper * self.time_step / 5.0 * self.transfer_upper * self.time_step / \
                    5.0 * \
                    (d_tempatmo_d_atmoconc[i - 1, j] -
                     d_tempocean_d_atmoconc[i - 1, j])
                #-------ocean temp derivative-----------
                # if atmo temp is saturated
                if (self.temperature_df.at[years[i], 'temp_atmo'] == self.up_tatmo):
                    d_tempatmo_d_atmoconc[i, j] = 0

                d_tempocean_d_atmoconc[i, j] = d_tempocean_d_atmoconc[i - 1, j] \
                    + self.transfer_lower * self.time_step / 5.0 * \
                    (d_tempatmo_d_atmoconc[i - 1, j] -
                     d_tempocean_d_atmoconc[i - 1, j])
                j = j + 1

        return d_tempatmo_d_atmoconc / self.scale_factor_carbon_cycle, d_tempocean_d_atmoconc / self.scale_factor_carbon_cycle

    def compute_d_temp_atmo_objective(self):
        """ Compute derivative of temperature objective function regarding atmospheric temperature
        """

        temperature_df_values = self.temperature_df['temp_atmo'].values
        delta_years = len(temperature_df_values)
        result = np.zeros(len(temperature_df_values))

        if self.temperature_obj_option == TempChange.LAST_TEMPERATURE_OBJECTIVE:
            # (1-alpha) => C
            # self.temperature_df['temp_atmo'][-1] => xf
            # self.temperature_df['temp_atmo'][0]) => x1

            # C * xf / x1

            # derivative at n=1
            # -(C * xf) / x1²
            #
            # derivative at 1 < n < f
            # 0
            #
            # derivative at n = f
            # C / x1

            result[0] = (1 - self.beta) * (1 - self.alpha) / \
                self.temperature_change_ref
            result[-1] = (1 - self.beta) * (1 - self.alpha) / \
                self.temperature_change_ref

        elif self.temperature_obj_option == TempChange.INTEGRAL_OBJECTIVE:
            # (1-alpha) => C
            # self.temperature_df['temp_atmo'].sum() => (x1 + x2 + ... + xn)
            # self.temperature_df['temp_atmo'][0] * delta_years) => x1*W

            # C(x1 + x2 + .. + xn) / (x1 * W)

            # derivative at n=1
            # -((x2 + ... + xn)*C) / W * x1²
            #
            # derivative at n > 1
            # C / x1W

            #             dn1 = -1.0 * ((1 - self.beta) * (1 - self.alpha) * (self.temperature_df['temp_atmo'].sum() - temperature_df_values[0])) / (
            #                 (temperature_df_values[0] ** 2) * delta_years)

            dnn = (1 - self.beta) * (1 - self.alpha) / \
                (self.temperature_change_ref * delta_years)

            for index in range(len(temperature_df_values)):
                if index != 0:
                    result[index] = dnn
        else:
            raise ValueError(
                f'temperature_obj_option = "{self.temperature_obj_option}" is not one of the allowed value : {TempChange.LAST_TEMPERATURE_OBJECTIVE} or {TempChange.INTEGRAL_OBJECTIVE}')

        return result

    def compute(self, in_dict):
        """
        Compute all
        """
        # rescale atmo_conc of carbon_cycle_df
        self.carboncycle_df = pd.DataFrame({'years': in_dict['carboncycle_df']['years'].values,
                                            'atmo_conc': in_dict['carboncycle_df']['atmo_conc'].values /
                                            self.scale_factor_carbon_cycle})
        self.carboncycle_df.index = self.carboncycle_df['years'].values

        self.compute_forcing()

        for year in self.years_range[1:]:
            self.compute_temp_atmo(year)
            self.compute_temp_ocean(year)

        self.temperature_df = self.temperature_df.replace(
            [np.inf, -np.inf], np.nan)

        #-- Compute temperature objectives with alpha trades and beta weight with CO2 objective
        temperature_df_values = self.temperature_df['temp_atmo'].values
        delta_years = len(temperature_df_values)

        if self.temperature_obj_option == TempChange.LAST_TEMPERATURE_OBJECTIVE:
            self.temperature_objective = np.asarray([(1 - self.beta) * (1 - self.alpha) * (temperature_df_values[-1])
                                                     / (self.temperature_change_ref)])

        elif self.temperature_obj_option == TempChange.INTEGRAL_OBJECTIVE:
            self.temperature_objective = np.asarray([(1 - self.beta) * (1 - self.alpha) * self.temperature_df['temp_atmo'].sum()
                                                     / (self.temperature_change_ref * delta_years)])
        else:
            raise ValueError(
                f'temperature_obj_option = "{self.temperature_obj_option}" is not one of the allowed value : {TempChange.LAST_TEMPERATURE_OBJECTIVE} or {TempChange.INTEGRAL_OBJECTIVE}')

        return self.temperature_df.fillna(0.0), self.temperature_objective
