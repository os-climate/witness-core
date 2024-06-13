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


class UtilityModel():
    '''
    Used to compute population welfare and utility
    '''

    def __init__(self, param):
        '''
        Constructor
        '''
        self.shift_scurve = 0.
        self.strech_scurve = 0.
        self.discounted_utility_quantity_objective = 0.
        self.param = param

        self.set_data()

        self.n_years = None

        self.economics_df = None
        self.energy_mean_price = None
        self.population_df = None
        self.create_dataframe()

    def set_data(self):
        self.year_start = self.param[GlossaryCore.YearStart]
        self.year_end = self.param[GlossaryCore.YearEnd]
        self.time_step = self.param[GlossaryCore.TimeStep]  # time_step

        self.shift_scurve = self.param['shift_scurve']
        self.strech_scurve = self.param['strech_scurve']

        self.conso_elasticity = self.param['conso_elasticity']  # elasmu
        self.init_rate_time_pref = self.param['init_rate_time_pref']  # prstp
        self.initial_raw_energy_price = self.param['initial_raw_energy_price']
        self.init_discounted_utility = self.param['init_discounted_utility']

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        years_range = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years_range = years_range
        self.n_years = len(self.years_range)
        utility_df = pd.DataFrame(
            index=years_range,
            columns=GlossaryCore.UtilityDf['dataframe_descriptor'].keys())

        for key in utility_df.keys():
            utility_df[key] = 0
        utility_df[GlossaryCore.Years] = years_range
        self.utility_df = utility_df
        return utility_df

    def compute_utility_discount_rate(self):
        """
        Compute Average utility social discount rate
         rr(t) = 1/((1+prstp)**(tstep*(t.val-1)));
        """
        t = ((self.years_range - self.year_start) / self.time_step) + 1
        u_discount_rate = 1 / ((1 + self.init_rate_time_pref)
                               ** (self.time_step * (t - 1)))
        self.utility_df[GlossaryCore.UtilityDiscountRate] = u_discount_rate
        return u_discount_rate


    def compute(self, economics_df, energy_mean_price, population_df):
        """compute"""
        self.economics_df = economics_df
        self.energy_mean_price = energy_mean_price
        self.energy_price_ref = self.initial_raw_energy_price
        self.population_df = population_df

        self.compute_utility_discount_rate()
        self.compute_utility_quantity()
        self.compute_discounted_utility_quantity()
        self.compute_quantity_objective()

        return self.utility_df

    def compute_utility_quantity(self):
        """
        Consumption = Quantity (of "things" consumed") * Price ("average price of things consumed")

        We consider that the average price of things that are consumed is driven by energy price.
        We want to maximize the quantity of things consumed,

        quantity = consumption / price

        If we take year start as a reference point ("1")

        utility quantity (year) = quantity (year) / quantity (year start)
        """
        consumption = self.economics_df[GlossaryCore.PerCapitaConsumption].values
        consumption_year_start = consumption[0]

        energy_price = self.energy_mean_price[GlossaryCore.EnergyPriceValue].values

        quantity_year_start = consumption_year_start / self.energy_price_ref
        quantity = consumption / energy_price

        utility_quantity = quantity / quantity_year_start

        self.utility_df[GlossaryCore.UtilityQuantity] = self.s_curve_function(utility_quantity)

    def s_curve_function(self, x):
        y = (x - 1 - self.shift_scurve) * self.strech_scurve
        return 1 / (1 + np.exp(-y))

    def d_s_curve_function(self, x):
        u_prime = self.strech_scurve
        u = (x - 1 - self.shift_scurve) * self.strech_scurve
        f_prime_u = np.exp(-u) / (1+np.exp(-u)) ** 2
        return u_prime * f_prime_u


    def compute_discounted_utility_quantity(self):
        """
        Discounted utility quantity (year) = Utility quantity(year) * discount factor (year)
        """
        utility_quantity = self.utility_df[GlossaryCore.UtilityQuantity].values
        discount_factor = self.utility_df[GlossaryCore.UtilityDiscountRate].values

        self.utility_df[GlossaryCore.DiscountedUtilityQuantity] = utility_quantity * discount_factor

    def compute_quantity_objective(self):
        """Quantity objective = Mean over years of discounted utility quantity"""
        quantity_obj_val = self.utility_df[GlossaryCore.DiscountedUtilityQuantity].values.mean()
        self.discounted_utility_quantity_objective = np.array([quantity_obj_val])


    ######### GRADIENTS ########
    def d_utility_quantity(self):
        energy_price = self.energy_mean_price[GlossaryCore.EnergyPriceValue].values
        pcc = self.economics_df[GlossaryCore.PerCapitaConsumption].values
        u = (pcc / energy_price) / (pcc[0] / self.energy_price_ref)
        d_u_d_ep = -(pcc / energy_price ** 2) / (pcc[0] / self.energy_price_ref)
        d_u_d_pcc = (1 / energy_price) / (pcc[0] / self.energy_price_ref)
        d_u_d_pcc0 = -(pcc / energy_price) / (pcc[0] ** 2 / self.energy_price_ref)

        d_utility_denergy_price = np.diag(d_u_d_ep * self.d_s_curve_function(u))
        d_utility_dpcc = np.diag(d_u_d_pcc * self.d_s_curve_function(u))
        d_utility_dpcc0 = d_u_d_pcc0 * self.d_s_curve_function(u)
        d_utility_dpcc[:, 0] = d_utility_dpcc0
        d_utility_dpcc[0,0] = 0.
        discount_rate = self.utility_df[GlossaryCore.UtilityDiscountRate].values
        d_discounted_utility_quantity_denergy_price = np.diag(discount_rate) * d_utility_denergy_price
        d_discounted_utility_quantity_dpcc = np.diag(discount_rate) @ d_utility_dpcc

        d_utility_obj_d_energy_price = d_discounted_utility_quantity_denergy_price.mean(axis=0)
        d_utility_obj_dpcc = d_discounted_utility_quantity_dpcc.mean(axis=0)
        return d_utility_denergy_price, d_utility_dpcc,\
               d_discounted_utility_quantity_denergy_price, d_discounted_utility_quantity_dpcc,\
               d_utility_obj_d_energy_price, d_utility_obj_dpcc
