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
        self.discounted_utility_quantity_objective = 0.
        self.param = param
        self.per_capita_consumption_ref = None

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

        self.conso_elasticity = self.param['conso_elasticity']  # elasmu
        self.init_rate_time_pref = self.param['init_rate_time_pref']  # prstp
        self.initial_raw_energy_price = self.param['initial_raw_energy_price']
        self.init_discounted_utility = self.param['init_discounted_utility']
        self.per_capita_consumption_ref = self.param[GlossaryCore.PerCapitaConsumptionUtilityRefName]
        #self.min_period_utility = 0.01²

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

    def compute_energy_price_ratio(self):
        """energy price ratio is energy_price_ref/energy_price"""
        energy_price = self.energy_mean_price[GlossaryCore.EnergyPriceValue].values
        energy_price_ratio = self.energy_price_ref / energy_price
        self.utility_df[GlossaryCore.EnergyPriceRatio] = energy_price_ratio

    def compute_per_capita_consumption_utility(self):
        """Per capita consumption utilty is ((percapitaconso**(1-elasmu)-1)/(1-elasmu)-1)
        leads to <0 values at small pc_consumption values => keep pc_consumption only"""
        pc_consumption = self.economics_df[GlossaryCore.PerCapitaConsumption].values
        #consumption_utility = (pc_consumption ** (1 - self.conso_elasticity) - 1) / (1 - self.conso_elasticity) - 1
        self.utility_df[GlossaryCore.PerCapitaConsumptionUtility] = pc_consumption

    def compute_utility(self):
        """
        Utility = Energy price ratio * PerCapitaUtilityOfConsumption
        """
        energy_price_ratio = self.utility_df[GlossaryCore.EnergyPriceRatio].values
        consumption_utility = self.utility_df[GlossaryCore.PerCapitaConsumptionUtility].values
        adjusted_period_utility = consumption_utility * energy_price_ratio
        self.utility_df[GlossaryCore.PeriodUtilityPerCapita] = adjusted_period_utility

    def compute_discounted_utility(self):
        """
        period Utility
        PERIODU_pc(t) * rr(t) * L(t)
        """
        utility = self.utility_df[GlossaryCore.PeriodUtilityPerCapita].values
        u_discount_rate = self.utility_df[GlossaryCore.UtilityDiscountRate].values
        population = self.population_df[GlossaryCore.PopulationValue].values
        discounted_utility = utility * u_discount_rate * population
        self.utility_df[GlossaryCore.DiscountedUtility] = discounted_utility
        return discounted_utility

    def compute_normalized_welfare(self):  # rescalenose
        """Normalized Welfare = sum of discounted utility / n_years / init discounted utility"""

        self.normalized_welfare = np.asarray([self.utility_df[GlossaryCore.DiscountedUtility].sum()]) / self.n_years / self.init_discounted_utility

    def compute_negative_welfare_objective(self):
        """
        Compute negative welfare objective as - welfare / init_discounted_utility * n_years
        """
        self.negative_welfare_objective = -1.*  self.normalized_welfare

    def compute_inverse_welfare_objective(self):
        self.inverse_welfare_objective = 1. / self.normalized_welfare

    def compute_negative_last_year_utility_objective(self):
        """objective for utility at last year"""
        last_year_discounted_utility = self.utility_df[GlossaryCore.DiscountedUtility][self.year_end]
        self.last_year_utility_objective = np.asarray([- last_year_discounted_utility / self.init_discounted_utility])

    def compute_per_capita_consumption_utility_objective(self):
        """
        Objective for capita consumption (without energy price effect)
        """
        self.per_capita_consumption_objective = -1.0 * (np.asarray([self.utility_df[GlossaryCore.PerCapitaConsumptionUtility].sum()])
                                                 / (self.n_years * self.per_capita_consumption_ref))

    ######### GRADIENTS ########

    def d_energy_price_ratio_d_energy_price(self):
        energy_price = self.energy_mean_price[GlossaryCore.EnergyPriceValue].values
        return np.diag(-1.* self.energy_price_ref / energy_price ** 2)

    def d_utility_d_energy_price(self):
        """utility = per capita consumption utility * energy price ratio"""
        utility = self.utility_df[GlossaryCore.PerCapitaConsumptionUtility].values
        d_energy_price_ratio_d_energy_price = self.d_energy_price_ratio_d_energy_price()
        return d_energy_price_ratio_d_energy_price * utility

    def d_pc_consumption_utility_d_per_capita_consumption(self):
        """pass"""
        pc_consumption = self.economics_df[GlossaryCore.PerCapitaConsumption].values
        #d_pc_consumption_utility_d_pc_consumption = pc_consumption ** (-self.conso_elasticity)
        #np.diag(d_pc_consumption_utility_d_pc_consumption)

        return np.identity(len(pc_consumption))

    def d_utility_d_per_capita_consumption(self):
        """utility = utility of per capita consumption * energy price ratio"""
        d_pc_consumption_utility_d_pc_consumption = self.d_pc_consumption_utility_d_per_capita_consumption()
        energy_price_ratio = self.utility_df[GlossaryCore.EnergyPriceRatio].values
        d_utility_d_par_capita_consumption = d_pc_consumption_utility_d_pc_consumption * energy_price_ratio
        return d_utility_d_par_capita_consumption

    def d_discounted_utility_d_population(self):
        """discounted utility = utility discount rate * utility * population"""

        u_discount_rate = self.utility_df[GlossaryCore.UtilityDiscountRate]
        utility = self.utility_df[GlossaryCore.PeriodUtilityPerCapita]
        d_discounted_utility_d_pop = np.diag(utility * u_discount_rate)
        return d_discounted_utility_d_pop

    def d_discounted_utility_d_user_input(self, d_utility_d_user_input):
        u_discount_rate = self.utility_df[GlossaryCore.UtilityDiscountRate]
        population = self.population_df[GlossaryCore.PopulationValue]
        d_discounted_utility_d_utility = np.diag(population * u_discount_rate)
        d_discounted_utility_d_user_input = d_discounted_utility_d_utility * d_utility_d_user_input
        return d_discounted_utility_d_user_input

    def d_normalized_welfare_d_user_input(self, d_discounted_utility_d_user_input):
        d_normalized_welfare_d_user_input = np.ones_like(self.years_range) / self.init_discounted_utility / self.n_years
        d_welfare_d_user_input = d_discounted_utility_d_user_input @ d_normalized_welfare_d_user_input
        return d_welfare_d_user_input

    def d_objectives_d_user_input(self, d_discounted_utility_d_user_input):
        d_normalized_welfare_d_user_input = self.d_normalized_welfare_d_user_input(d_discounted_utility_d_user_input)
        d_negative_welfare_d_normlized_welfare = -1
        d_negative_welfare_objective_d_user_input = d_negative_welfare_d_normlized_welfare * d_normalized_welfare_d_user_input

        d_inverse_welfare_d_welfare = - 1. / self.normalized_welfare ** 2
        d_inverse_welfare_objective_d_user_input = d_inverse_welfare_d_welfare * d_normalized_welfare_d_user_input

        return d_negative_welfare_objective_d_user_input, d_inverse_welfare_objective_d_user_input

    def d_last_utility_objective_d_user_input(self, d_discounted_utility_d_user_input):
        """Last utility objective = - discounted utility at year end / discounted utility at year start"""
        d_last_utility_d_discounted_utility = np.zeros_like(self.years_range, dtype=float)
        d_last_utility_d_discounted_utility[-1] = -1. / self.init_discounted_utility
        d_last_utility_objective_d_user_input = d_last_utility_d_discounted_utility @ d_discounted_utility_d_user_input
        return d_last_utility_objective_d_user_input

    def d_pc_consumption_utility_objective_d_per_capita_consumption(self):
        """derivative of consumption utility per capita objective wrt per capita consumption"""
        pc_consumption = self.economics_df[GlossaryCore.PerCapitaConsumption].values
        d_pc_consumption_utility_objective_d_pc_consumption = -1.0 * np.sum(np.identity(len(pc_consumption)),
                                                                     axis=0) / (self.n_years * self.per_capita_consumption_ref)
        return d_pc_consumption_utility_objective_d_pc_consumption


    def compute(self, economics_df, energy_mean_price, population_df):
        """compute"""
        self.economics_df = economics_df
        self.energy_mean_price = energy_mean_price
        self.energy_price_ref = self.initial_raw_energy_price
        self.population_df = population_df

        self.compute_utility_discount_rate()
        self.compute_energy_price_ratio()
        self.compute_utility_quantity()
        self.compute_per_capita_consumption_utility()
        self.compute_utility()
        self.compute_discounted_utility()
        self.compute_normalized_welfare()
        self.compute_negative_welfare_objective()
        self.compute_inverse_welfare_objective()
        self.compute_discounted_utility_quantity()
        self.compute_quantity_objective()
        self.compute_negative_last_year_utility_objective()
        self.compute_per_capita_consumption_utility_objective()

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

        self.utility_df[GlossaryCore.UtilityQuantity] = np.log(utility_quantity)

    def compute_discounted_utility_quantity(self):
        """
        Discounted utility quantity (year) = Utility quantity(year) * discount factor (year)
        """
        utility_quantity = self.utility_df[GlossaryCore.UtilityQuantity].values
        discount_factor = self.utility_df[GlossaryCore.UtilityDiscountRate].values

        self.utility_df[GlossaryCore.DiscountedUtilityQuantity] = utility_quantity * discount_factor

    def compute_quantity_objective(self):
        """Quantity objective = Mean over years of discounted utility quantity"""
        self.discounted_utility_quantity_objective = np.array([self.utility_df[GlossaryCore.DiscountedUtilityQuantity].values.mean()])

    def d_utility_quantity(self):
        energy_price = self.energy_mean_price[GlossaryCore.EnergyPriceValue].values
        pcc = self.economics_df[GlossaryCore.PerCapitaConsumption].values
        d_utility_denergy_price = np.diag(- 1 / energy_price)
        d_utility_dpcc = np.diag(1 / pcc)
        d_utility_dpcc[:, 0] = - 1 / pcc[0]
        d_utility_dpcc[0,0] = 0.
        discount_rate = self.utility_df[GlossaryCore.UtilityDiscountRate].values
        d_discounted_utility_quantity_denergy_price = np.diag(discount_rate) * d_utility_denergy_price
        d_discounted_utility_quantity_dpcc = np.diag(discount_rate) @ d_utility_dpcc

        d_utility_obj_d_energy_price = d_discounted_utility_quantity_denergy_price.mean(axis=0)
        d_utility_obj_dpcc = d_discounted_utility_quantity_dpcc.mean(axis=0)
        return d_utility_denergy_price, d_utility_dpcc,\
               d_discounted_utility_quantity_denergy_price, d_discounted_utility_quantity_dpcc,\
               d_utility_obj_d_energy_price, d_utility_obj_dpcc
