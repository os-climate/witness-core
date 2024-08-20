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
from pandas.core.frame import DataFrame

from climateeconomics.glossarycore import GlossaryCore


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
        self.ghg_cycle_df = None
        self.forcing_df = None
        self.temperature_objective = None
        self.temperature_end_constraint = None
        self.ppm_to_gtc = 2.13
        self.set_data(inputs)
        self.create_dataframe()

    def set_data(self, inputs):
        self.year_start = inputs[GlossaryCore.YearStart]
        self.year_end = inputs[GlossaryCore.YearEnd]
        self.time_step = inputs[GlossaryCore.TimeStep]
        self.init_temp_ocean = inputs['init_temp_ocean']
        self.init_temp_atmo = inputs['init_temp_atmo']
        self.eq_temp_impact = inputs['eq_temp_impact']
        self.temperature_model = inputs['temperature_model']
        self.forcing_model = inputs['forcing_model']
        self.ghg_cycle_df = inputs[GlossaryCore.GHGCycleDfValue]

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

        self.temperature_change_ref = inputs['temperature_change_ref']

        self.temperature_end_constraint_limit = inputs['temperature_end_constraint_limit']
        self.temperature_end_constraint_ref = inputs['temperature_end_constraint_ref']

        # FUND
        self.climate_sensitivity = 3.0

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        years_range = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years_range = years_range
        self.temperature_df = DataFrame({GlossaryCore.Years: self.years_range})
        self.forcing_df = DataFrame({GlossaryCore.Years: self.years_range})

    def compute_exog_forcing_dice(self):
        """
        Compute exogenous forcing for other greenhouse gases following DICE pyworld3
        linear increase of exogenous forcing following a given scenario
        """

        exog_forcing = np.linspace(
            self.init_forcing_nonco, self.hundred_forcing_nonco, len(self.years_range))

        self.forcing_df['CH4 and N20 forcing'] = exog_forcing
        return exog_forcing

    def compute_exog_forcing_myhre(self, ch4_ppm, n2o_ppm):
        """
        Compute exogenous forcing for CH4 and N2O gases following Myhre pyworld3
        Myhre et al, 1998, JGR, doi: 10.1029/98GL01908
        Myhre pyworld3 can be found in FUND, MAGICC and FAIR IAM models

        in FUND 0.036 * 1.4(np.sqrt(ch4_conc) - np.sqrt(ch4_conc_init))
        in FAIR 0.036 (np.sqrt(ch4_conc) -*np.sqrt(ch4_conc_init))

        We use ppm unit to compute this forcing as in FAIR and FUND
        """

        def MN(c1, c2):
            return 0.47 * np.log(1 + 2.01e-5 * (c1 * c2)**(0.75) +
                                 5.31e-15 * c1 * (c1 * c2)**(1.52))

        exog_forcing = 0.036 * (np.sqrt(ch4_ppm) - np.sqrt(self.ch4_conc_init_ppm)) - MN(ch4_ppm, self.n2o_conc_init_ppm) + \
            0.12 * (np.sqrt(n2o_ppm) - np.sqrt(self.n2o_conc_init_ppm)) - MN(self.ch4_conc_init_ppm, n2o_ppm) \
            + 2 * MN(self.ch4_conc_init_ppm, self.n2o_conc_init_ppm)

        self.forcing_df['CH4 and N2O forcing'] = exog_forcing
        return exog_forcing

    def compute_forcing_etminan(self, co2_ppm, ch4_ppm, n2o_ppm):
        """
        Compute radiative forcing following Etminan pyworld3 (found in FAIR)
        Etminan, M., Myhre, G., Highwood, E., and Shine, K.: Radiative forcing of carbon dioxide, methane, and nitrous oxide: A
        significant revision of the methane radiative forcing, Geophysical Research Letters, 43, 2016.
        """

        co2mean = 0.5 * (co2_ppm + self.c0_ppm)
        ch4mean = 0.5 * (ch4_ppm + self.ch4_conc_init_ppm)
        n2omean = 0.5 * (n2o_ppm + self.n2o_conc_init_ppm)

        # sign values instead of fabs because gradients are not well computed
        # with np.abs
        sign_values = np.ones(len(co2_ppm))
        sign_values[co2_ppm.real < self.c0_ppm] = -1

        co2_forcing = (-2.4e-7 * (co2_ppm - self.c0_ppm)**2 + 7.2e-4 * sign_values * (co2_ppm - self.c0_ppm) -
                       2.1e-4 * n2omean + self.forcing_eq_co2 / np.log(2)) * np.log(co2_ppm / self.c0_ppm)
        ch4_forcing = (-1.3e-6 * ch4mean - 8.2e-6 * n2omean + 0.043) * (np.sqrt(ch4_ppm) -
                                                                        np.sqrt(self.ch4_conc_init_ppm))
        n2o_forcing = (-8.0e-6 * co2mean + 4.2e-6 * n2omean - 4.9e-6 * ch4mean + 0.117) * \
            (np.sqrt(n2o_ppm) - np.sqrt(self.n2o_conc_init_ppm))

        self.forcing_df['CO2 forcing'] = co2_forcing
        self.forcing_df['CH4 forcing'] = ch4_forcing
        self.forcing_df['N2O forcing'] = n2o_forcing

        return co2_forcing + ch4_forcing + n2o_forcing

    def compute_forcing_meinshausen(self, co2_ppm, ch4_ppm, n2o_ppm):
        """
        Compute radiative forcing following MeinsHausen pyworld3 (found in FAIR)
        Meinshausen, M., Nicholls, Z.R., Lewis, J., Gidden, M.J., Vogel, E., Freund, 
        M., Beyerle, U., Gessner, C., Nauels, A., Bauer, N. and Canadell, J.G., 2020.
        The shared socio-economic pathway (SSP) greenhouse gas concentrations and their extensions to 2500. 
        Geoscientific Model Development, 13(8), pp.3571-3605.
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

        Camax = self.c0_ppm - b1 / (2 * a1)
        # if self.c0_ppm < atmo_conc_ppm <= Camax:  # the most likely case
        alphap = d1 + a1 * (co2_ppm - self.c0_ppm)**2 + \
            b1 * (co2_ppm - self.c0_ppm)

        alphap[co2_ppm <= self.c0_ppm] = d1
        alphap[co2_ppm >= Camax] = d1 - b1**2 / (4 * a1)

        alpha_n2o = c1 * np.sqrt(n2o_ppm)
        co2_forcing = (alphap + alpha_n2o) * np.log(co2_ppm / self.c0_ppm)
        self.forcing_df['CO2 forcing'] = co2_forcing

        # CH4
        ch4_forcing = (
            a3 * np.sqrt(ch4_ppm) + b3 * np.sqrt(n2o_ppm) + d3) * (np.sqrt(ch4_ppm) - np.sqrt(self.ch4_conc_init_ppm))

        self.forcing_df['CH4 forcing'] = ch4_forcing
        # N2O
        n2o_forcing = (a2 * np.sqrt(co2_ppm) + b2 * np.sqrt(n2o_ppm) +
                       c2 * np.sqrt(ch4_ppm) + d2) * (np.sqrt(n2o_ppm) - np.sqrt(self.n2o_conc_init_ppm))

        self.forcing_df['N2O forcing'] = n2o_forcing

        return co2_forcing + ch4_forcing + n2o_forcing

    def compute_log_co2_forcing(self, co2_ppm):
        co2_forcing = self.forcing_eq_co2 / np.log(2) * \
            np.log(co2_ppm / (self.c0_ppm))
        return co2_forcing

    def compute_forcing(self):
        """
        Compute increase in radiative forcing for t using values at t-1
        (watts per m2 from 1900)
        """
        co2_ppm = self.ghg_cycle_df[GlossaryCore.CO2Concentration].values
        ch4_ppm = self.ghg_cycle_df[GlossaryCore.CH4Concentration].values
        n2o_ppm = self.ghg_cycle_df[GlossaryCore.N2OConcentration].values

        if self.forcing_model == 'DICE':

            exog_forcing = self.compute_exog_forcing_dice()
            co2_forcing = self.compute_log_co2_forcing(co2_ppm)
            self.forcing_df['CO2 forcing'] = co2_forcing
            forcing = co2_forcing + exog_forcing

        elif self.forcing_model == 'Myhre':
            exog_forcing = self.compute_exog_forcing_myhre(ch4_ppm, n2o_ppm)
            co2_forcing = self.compute_log_co2_forcing(co2_ppm)
            self.forcing_df['CO2 forcing'] = co2_forcing
            forcing = co2_forcing + exog_forcing

        elif self.forcing_model == 'Etminan':

            forcing = self.compute_forcing_etminan(co2_ppm, ch4_ppm, n2o_ppm)

        elif self.forcing_model == 'Meinshausen':

            forcing = self.compute_forcing_meinshausen(co2_ppm, ch4_ppm, n2o_ppm)

        else:

            raise Exception("forcing model not in available models")

        self.temperature_df[GlossaryCore.Forcing] = forcing

    ######### DICE ########
    def compute_temp_atmo_ocean_dice(self):
        """
        Compute temperature of atmosphere (t) using t-1 values
        """
        temp_atmo = self.init_temp_atmo
        temp_ocean = self.init_temp_ocean

        temp_atmo_list = [temp_atmo]
        temp_ocean_list = [temp_ocean]

        for year, forcing in zip(self.years_range[1:], self.temperature_df[GlossaryCore.Forcing].values[1:]) :
            new_temp_ocean = temp_ocean + (self.transfer_lower / (5.0 / self.time_step)) * (temp_atmo - temp_ocean)
            new_temp_atmo = temp_atmo + (self.climate_upper / (5.0 / self.time_step)) * \
                ((forcing - (self.forcing_eq_co2 / self.eq_temp_impact) *
                  temp_atmo) - ((self.transfer_upper / (5.0 / self.time_step)) * (temp_atmo - temp_ocean)))

            temp_atmo = new_temp_atmo
            temp_ocean = new_temp_ocean

            temp_atmo_list.append(temp_atmo)
            temp_ocean_list.append(temp_ocean)

        temp_atmo_list = np.array(temp_atmo_list)
        temp_ocean_list = np.array(temp_ocean_list)

        temp_ocean_list = np.maximum(temp_ocean_list, self.lo_tocean)
        temp_ocean_list = np.minimum(temp_ocean_list, self.up_tocean)
        temp_atmo_list = np.minimum(temp_atmo_list, self.up_tatmo)

        self.temperature_df[GlossaryCore.TempOcean] = temp_ocean_list
        self.temperature_df[GlossaryCore.TempAtmo] = temp_atmo_list

    ######### FUND ########
    def compute_temp_fund(self):
        """
        Compute temperature of atmosphere (t) using t-1 values following FUND Model
        """
        alpha = -42.7
        beta_l = 29.1
        beta_q = 0.001
        cs = self.climate_sensitivity
        e_folding_time = max(alpha +
                             beta_l * cs +
                             beta_q * cs * cs,
                             1)
        temperature = self.init_temp_atmo
        temperature_list = [temperature]

        for year, radiative_forcing in zip(self.years_range[1:], self.temperature_df[GlossaryCore.Forcing].values[1:]):
            temperature = (1-1/e_folding_time) * temperature + cs/(5.35*np.log(2)*e_folding_time) * radiative_forcing
            temperature_list.append(temperature)

        self.temperature_df[GlossaryCore.TempAtmo] = temperature_list

    def compute_sea_level_fund(self):
        """
        Compute seal level (t) using t-1 values following FUND Model
        """
        rho = 500
        gamma = 2
        initial_sea_level = 0.
        temp_atmo = self.temperature_df[GlossaryCore.TempAtmo].values
        sea_level = (1 - 1 / rho) * initial_sea_level + gamma * temp_atmo / rho

        self.temperature_df['sea_level'] = sea_level

    ######### CONSTRAINT ########
    def compute_temperature_year_end_constraint(self):
        """
        Compute temperature constraint
        """
        temp_atmo_year_end = self.temperature_df[GlossaryCore.TempAtmo].values[-1]
        self.temperature_end_constraint = np.array([(self.temperature_end_constraint_limit - temp_atmo_year_end)/self.temperature_end_constraint_ref])

    ######### GRADIENTS ########

    def compute_d_forcing(self):
        """
        Compute gradient for radiative forcing 
        """

        co2_ppm = self.ghg_cycle_df[GlossaryCore.CO2Concentration].values
        ch4_ppm = self.ghg_cycle_df[GlossaryCore.CH4Concentration].values
        n2o_ppm = self.ghg_cycle_df[GlossaryCore.N2OConcentration].values

        if self.forcing_model == 'DICE':
            dco2_forcing = self.compute_dlog_co2_forcing(co2_ppm)
            dforcing = dco2_forcing
            self.d_forcing_datmo_conc_dict = {
                'CO2 forcing': dforcing}

        elif self.forcing_model == 'Myhre':
            dexog_forcing_ch4, dexog_forcing_n2o = self.compute_dexog_forcing_myhre(ch4_ppm, n2o_ppm)
            dco2_forcing = self.compute_dlog_co2_forcing(co2_ppm)
            self.d_forcing_datmo_conc_dict = {'CO2 forcing': dco2_forcing,
                                              'CH4 forcing': dexog_forcing_ch4,
                                              'N2O forcing': dexog_forcing_n2o}
            dforcing = dco2_forcing + dexog_forcing_ch4 + dexog_forcing_n2o

        elif self.forcing_model == 'Etminan':
            dforcing = self.compute_dforcing_etminan(co2_ppm, ch4_ppm, n2o_ppm)

        elif self.forcing_model == 'Meinshausen':

            dforcing = self.compute_dforcing_meinshausen(co2_ppm, ch4_ppm, n2o_ppm)

        else:
            raise Exception("forcing model not in available models")
        return dforcing

    def compute_dlog_co2_forcing(self, co2_ppm):

        d_forcing = self.forcing_eq_co2 / (np.log(2) * co2_ppm)

        return d_forcing

    def compute_dexog_forcing_myhre(self, ch4_ppm, n2o_ppm):
        """
        Compute gradient for exogenous forcing for CH4 and N2O gases following Myhre pyworld3
        Myhre et al, 1998, JGR, doi: 10.1029/98GL01908
        """

        def dMN_dc1c2(c1, c2, x):

            if x == 'c1':
                f = 1 + 2.01e-5 * (c1 * c2)**(0.75) + \
                    5.31e-15 * c1 * (c1 * c2)**(1.52)
                fprime = 2.01e-5 * c2**(0.75) * 0.75 * c1**(0.75 - 1) + \
                    5.31e-15 * c2**(1.52) * 2.52 * c1**(2.52 - 1)
            elif x == 'c2':

                f = 1 + 2.01e-5 * (c1 * c2)**(0.75) + \
                    5.31e-15 * c1 * (c1 * c2)**(1.52)
                fprime = 2.01e-5 * c1**(0.75) * 0.75 * c2**(0.75 - 1) + \
                    5.31e-15 * c1**(2.52) * 1.52 * c2**(1.52 - 1)
            else:
                raise Exception("parameter of the method not in the possible list")

            return 0.47 * fprime / f

        dexog_forcing_ch4 = 0.036 / (2 * np.sqrt(ch4_ppm)) - \
            dMN_dc1c2(ch4_ppm, self.n2o_conc_init_ppm, 'c1')\

        dexog_forcing_n2o = 0.12 / (2 * np.sqrt(n2o_ppm)) - \
            dMN_dc1c2(self.ch4_conc_init_ppm, n2o_ppm, 'c2')

        return dexog_forcing_ch4, dexog_forcing_n2o

    def compute_dforcing_etminan(self, co2_ppm, ch4_ppm, n2o_ppm):
        """
        Compute gradient for Etminan pyworld3
        """
        co2mean = 0.5 * (co2_ppm + self.c0_ppm)
        ch4mean = 0.5 * (ch4_ppm + self.ch4_conc_init_ppm)
        n2omean = 0.5 * (n2o_ppm + self.n2o_conc_init_ppm)
        sign_values = np.ones(len(co2_ppm))
        sign_values[co2_ppm.real < self.c0_ppm] = -1

        # CO2
        dco2_forcing_dco2_ppm = (-2.4e-7 * 2.0 * (co2_ppm - self.c0_ppm) + sign_values * 7.2e-4) * np.log(co2_ppm / self.c0_ppm) + \
            1.0 / co2_ppm * (-2.4e-7 * (co2_ppm - self.c0_ppm)**2 + sign_values * 7.2e-4 * (co2_ppm - self.c0_ppm) -
                                              2.1e-4 * n2omean + self.forcing_eq_co2 / np.log(2))
        dco2_forcing_dn2o_ppm = -2.1e-4 * 0.5 * np.log(co2_ppm / self.c0_ppm)

        # CH4
        dch4_forcing_dch4_ppm = (-1.3e-6 * 0.5) * (np.sqrt(ch4_ppm) - np.sqrt(self.ch4_conc_init_ppm)) +\
            (-1.3e-6 * ch4mean - 8.2e-6 * n2omean + 0.043) * 1.0 / (2.0 * np.sqrt(ch4_ppm))
        dch4_forcing_dn2o_ppm = (-8.2e-6 * 0.5) * (np.sqrt(ch4_ppm) - np.sqrt(self.ch4_conc_init_ppm))

        # N2O
        dn2o_forcing_dn2o_ppm = (4.2e-6 * 0.5) * (np.sqrt(n2o_ppm) - np.sqrt(self.n2o_conc_init_ppm)) + \
            (-8.0e-6 * co2mean + 4.2e-6 * n2omean - 4.9e-6 * ch4mean + 0.117) * 1.0 / (2.0 * np.sqrt(n2o_ppm))
        dn2o_forcing_dco2_ppm = (-8.0e-6 * 0.5) * (np.sqrt(n2o_ppm) - np.sqrt(self.n2o_conc_init_ppm))
        dn2o_forcing_dch4_ppm = (- 4.9e-6 * 0.5 ) * (np.sqrt(n2o_ppm) - np.sqrt(self.n2o_conc_init_ppm))

        dco2_forcing = dco2_forcing_dco2_ppm + dco2_forcing_dn2o_ppm
        dch4_forcing = dch4_forcing_dch4_ppm + dch4_forcing_dn2o_ppm
        dn2o_forcing = dn2o_forcing_dn2o_ppm + dn2o_forcing_dch4_ppm + dn2o_forcing_dco2_ppm

        self.d_forcing_datmo_conc_dict = {'CO2 forcing CO2 ppm': dco2_forcing_dco2_ppm,
                                          'CO2 forcing N2O ppm': dco2_forcing_dn2o_ppm,
                                          'CH4 forcing CH4 ppm': dch4_forcing_dch4_ppm,
                                          'CH4 forcing N2O ppm': dch4_forcing_dn2o_ppm,
                                          'N2O forcing CO2 ppm': dn2o_forcing_dco2_ppm,
                                          'N2O forcing CH4 ppm': dn2o_forcing_dch4_ppm,
                                          'N2O forcing N2O ppm': dn2o_forcing_dn2o_ppm,
                                          }
        return dco2_forcing + dch4_forcing + dn2o_forcing

    def compute_dforcing_meinshausen(self, co2_ppm, ch4_ppm, n2o_ppm):
        """
        Compute gradient for radiative forcing following MeinsHausen pyworld3

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

        Camax = self.c0_ppm - b1 / (2 * a1)
        # if self.c0_ppm < atmo_conc_ppm <= Camax:  # the most likely case
        alphap = d1 + a1 * (co2_ppm - self.c0_ppm)**2 + \
            b1 * (co2_ppm - self.c0_ppm)
        alphap[co2_ppm <= self.c0_ppm] = d1
        alphap[co2_ppm >= Camax] = d1 - b1**2 / (4 * a1)

        dalphap_dco2_ppm = a1 * 2.0 * (co2_ppm - self.c0_ppm) + b1
        dalphap_dco2_ppm[co2_ppm <= self.c0_ppm] = 0.0
        dalphap_dco2_ppm[co2_ppm >= Camax] = 0.0

        alpha_n2o = c1 * np.sqrt(n2o_ppm)
        alpha_n2o_dn2o_ppm = c1 / (2 * np.sqrt(n2o_ppm))

        # CO2 --> co2_forcing = (alphap + alpha_n2o) * np.log(co2_ppm / self.c0_ppm)
        dco2_forcing_dco2_ppm = (dalphap_dco2_ppm) * np.log(co2_ppm / self.c0_ppm) + \
            (alphap + alpha_n2o) / co2_ppm
        dco2_forcing_dn2o_ppm = (alpha_n2o_dn2o_ppm) * np.log(co2_ppm / self.c0_ppm)

        # CH4 --> ch4_forcing = (a3 * np.sqrt(ch4_ppm) + b3 * np.sqrt(n2o_ppm) + d3) * (np.sqrt(ch4_ppm) - np.sqrt(self.ch4_conc_init_ppm))
        dch4_forcing_dch4_ppm = a3 / (2.0 * np.sqrt(ch4_ppm)) * (np.sqrt(ch4_ppm) - np.sqrt(self.ch4_conc_init_ppm)) + \
                                (a3 * np.sqrt(ch4_ppm) + b3 * np.sqrt(n2o_ppm) + d3) / (2.0 * np.sqrt(ch4_ppm))

        dch4_forcing_dn2o_ppm = b3 / (2.0 * np.sqrt(n2o_ppm)) * (np.sqrt(ch4_ppm) - np.sqrt(self.ch4_conc_init_ppm))

        # N2O --> n2o_forcing = (a2 * np.sqrt(co2_ppm) + b2 * np.sqrt(n2o_ppm) + c2 * np.sqrt(ch4_ppm) + d2) * (np.sqrt(n2o_ppm) - np.sqrt(self.n2o_conc_init_ppm))
        dn2o_forcing_dco2_ppm = a2 / (2.0 * np.sqrt(co2_ppm)) * (np.sqrt(n2o_ppm) - np.sqrt(self.n2o_conc_init_ppm))

        dn2o_forcing_dch4_ppm = c2 / (2.0 * np.sqrt(ch4_ppm)) * (np.sqrt(n2o_ppm) - np.sqrt(self.n2o_conc_init_ppm))

        dn2o_forcing_dn2o_ppm = b2 / (2.0 * np.sqrt(n2o_ppm)) * (np.sqrt(n2o_ppm) - np.sqrt(self.n2o_conc_init_ppm)) + \
                       (a2 * np.sqrt(co2_ppm) + b2 * np.sqrt(n2o_ppm) + c2 * np.sqrt(ch4_ppm) + d2) / (2.0 * np.sqrt(n2o_ppm))

        dco2_forcing = dco2_forcing_dco2_ppm + dco2_forcing_dn2o_ppm
        dch4_forcing = dch4_forcing_dch4_ppm + dch4_forcing_dn2o_ppm
        dn2o_forcing = dn2o_forcing_dn2o_ppm + dn2o_forcing_dch4_ppm + dn2o_forcing_dco2_ppm

        self.d_forcing_datmo_conc_dict = {'CO2 forcing CO2 ppm': dco2_forcing_dco2_ppm,
                                          'CO2 forcing N2O ppm': dco2_forcing_dn2o_ppm,
                                          'CH4 forcing CH4 ppm': dch4_forcing_dch4_ppm,
                                          'CH4 forcing N2O ppm': dch4_forcing_dn2o_ppm,
                                          'N2O forcing CO2 ppm': dn2o_forcing_dco2_ppm,
                                          'N2O forcing CH4 ppm': dn2o_forcing_dch4_ppm,
                                          'N2O forcing N2O ppm': dn2o_forcing_dn2o_ppm,
                                          }
        return dco2_forcing + dch4_forcing + dn2o_forcing

    def compute_d_temp_atmo(self):

        nb_years = len(self.years_range)

        # derivative matrix initialization
        d_tempocean_d_atmoconc = np.zeros((nb_years, nb_years))

        # first line stays at zero since derivatives of initial values are zero
        # second line is only equal to the derivative of forcing effect
        dforcing_datmo_conc = self.compute_d_forcing()

        d_tempatmo_d_atmoconc = np.identity(nb_years) * self.climate_upper * self.time_step / 5.0 \
            * dforcing_datmo_conc

        d_tempatmo_d_atmoconc[0, 0] = 0.0

        for i in range(2, nb_years):
            j = 1

            # if temp_atmo is saturated at up_tatmo, it won't depend on atmo_conc anymore
            # so the derivative will be zero
            # if temp_ocean is saturated it has no effect as it only depends on
            # temp_atmo
            if self.temperature_df[GlossaryCore.TempAtmo].values[i] == self.up_tatmo:
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
                if self.temperature_df[GlossaryCore.TempAtmo].values[i] == self.up_tatmo:
                    d_tempatmo_d_atmoconc[i, j] = 0

                d_tempocean_d_atmoconc[i, j] = d_tempocean_d_atmoconc[i - 1, j] \
                    + self.transfer_lower * self.time_step / 5.0 * \
                    (d_tempatmo_d_atmoconc[i - 1, j] -
                     d_tempocean_d_atmoconc[i - 1, j])
                j = j + 1

        return d_tempatmo_d_atmoconc, d_tempocean_d_atmoconc

    def compute_d_temp_d_forcing_fund(self):
        """
        computes derivative of FUND temperature function
        """
        alpha = -42.7
        beta_l = 29.1
        beta_q = 0.001
        e_folding_time = max(alpha +
                             beta_l * self.climate_sensitivity +
                             beta_q * self.climate_sensitivity * self.climate_sensitivity,
                             1)

        coeff = self.climate_sensitivity/(5.35*np.log(2)*e_folding_time)
        decay = (1-1/e_folding_time)
        mat = np.diag(coeff*np.ones(len(self.years_range)))

        for i in np.arange(1, len(self.years_range-1)):
            coeff = coeff*decay
            mat += np.diag(coeff*np.ones(len(self.years_range)-i), -i)

        # first year is from initial data and is fixed ==> grad is zero
        mat[:, 0] = 0.0
        return mat

    def compute(self, in_dict) -> DataFrame:
        """
        Compute all
        """
        self.ghg_cycle_df = in_dict[GlossaryCore.GHGCycleDfValue]

        self.compute_forcing()

        if self.temperature_model == 'DICE':
            self.compute_temp_atmo_ocean_dice()

        elif self.temperature_model == 'FUND':

            self.compute_temp_fund()
            self.compute_sea_level_fund()

        elif self.temperature_model == 'FAIR':

            raise NotImplementedError("FAIR Not implemented yet")

        self.compute_temperature_year_end_constraint()
        return self.temperature_df.fillna(0.0)
