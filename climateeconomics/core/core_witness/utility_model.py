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

from climateeconomics.glossarycore import GlossaryCore


class UtilityModel():
    '''
    Used to compute population welfare and utility
    '''

    def __init__(self, param):
        '''
        Constructor
        '''
        self.param = param
        self.set_data()
        self.create_dataframe()

    def set_data(self):
        self.year_start = self.param['year_start']
        self.year_end = self.param['year_end']
        self.time_step = self.param['time_step']  # time_step

        self.conso_elasticity = self.param['conso_elasticity']  # elasmu
        self.init_rate_time_pref = self.param['init_rate_time_pref']  # prstp
        self.alpha = self.param['alpha']
        self.gamma = self.param['gamma']
        self.initial_raw_energy_price = self.param['initial_raw_energy_price']
        self.obj_option = self.param['welfare_obj_option']
        self.init_discounted_utility = self.param['init_discounted_utility']
        self.init_period_utility_pc = self.param['init_period_utility_pc']
        self.discounted_utility_ref = self.param['discounted_utility_ref']
        self.min_period_utility = 0.01

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        years_range = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years_range = years_range
        utility_df = pd.DataFrame(
            index=years_range,
            columns=[GlossaryCore.Years,
                     'u_discount_rate',
                     'period_utility_pc',
                     'discounted_utility',
                     'welfare'])

        for key in utility_df.keys():
            utility_df[key] = 0
        utility_df[GlossaryCore.Years] = years_range
        self.utility_df = utility_df
        return utility_df

    def compute__u_discount_rate(self, year):
        """
        Compute Average utility social discount rate
         rr(t) = 1/((1+prstp)**(tstep*(t.val-1)));
        """
        t = ((year - self.year_start) / self.time_step) + 1
        u_discount_rate = 1 / ((1 + self.init_rate_time_pref)
                               ** (self.time_step * (t - 1)))
        self.utility_df.loc[year, 'u_discount_rate'] = u_discount_rate
        return u_discount_rate

    def compute_period_utility(self, year):
        """
        Compute utility for period t per capita
         Args:
            :param consumption_pc: per capita consumption
                    energy_price: energy price 
                    energy_price_ref
             :type float

        ((percapitaconso**(1-elasmu)-1)/(1-elasmu)-1)*\
        energy_price_ref/energy_price   

        """

        # Compute energy price ratio
        energy_price = self.energy_mean_price.at[year, 'energy_price']
        energy_price_ratio = self.energy_price_ref / energy_price
        pc_consumption = self.economics_df.at[year, GlossaryCore.PerCapitaConsumption]
        period_utility = (
            pc_consumption**(1 - self.conso_elasticity) - 1) / (1 - self.conso_elasticity) - 1
        # need a limit for period utility because negative period utility is
        # not coherent and reverse gradient of utility vs energy price

        if period_utility < self.min_period_utility:
            period_utility = self.min_period_utility / 10.0 * \
                (9.0 + np.exp(
                    period_utility /
                    self.min_period_utility) * np.exp(-1))

        adjusted_period_utility = period_utility * energy_price_ratio
        self.utility_df.loc[year,
                            'period_utility_pc'] = adjusted_period_utility
        return period_utility

    def compute_discounted_utility(self, year):
        """
        period Utility
        PERIODU_pc(t) * rr(t) * L(t)
        """
        period_utility = self.utility_df.at[year, 'period_utility_pc']
        u_discount_rate = self.utility_df.at[year, 'u_discount_rate']
        population = self.population_df.at[year, GlossaryCore.PopulationValue]
        discounted_utility = period_utility * u_discount_rate * population
        self.utility_df.loc[year, 'discounted_utility'] = discounted_utility
        return discounted_utility

    def compute_welfare(self):  # rescalenose
        """
        Compute welfare
        tstep * scale1 * sum(t,  CEMUTOTPER(t)) + scale2
        """
        sum_u = sum(self.utility_df['discounted_utility'])
#         if rescale == True:
#             welfare = self.time_step * self.scaleone * sum_u + self.scaletwo
#         else:
#             welfare = self.time_step * self.scaleone * sum_u
#        return welfare
        self.utility_df.loc[self.year_end, 'welfare'] = sum_u
        return sum_u

    def compute_welfare_objective(self):
        """
        Objective function: inputs : alpha, gamma and obj_option
        """
        obj_option = self.obj_option

        n_years = len(self.years_range)
        if obj_option == 'last_utility':
            init_utility = self.init_period_utility_pc
            last_utility = self.utility_df.at[self.year_end,
                                              'period_utility_pc']
            welfare_objective = np.asarray(
                [self.alpha * init_utility / last_utility, ])
        elif obj_option == 'welfare':
            init_discounted_utility = self.init_discounted_utility
            welfare = self.utility_df['welfare'][self.year_end]
            # To avoid pb during convergence
            if welfare / (init_discounted_utility * n_years) < 0.01:
                welfare = 0.01 + \
                    np.exp(welfare / (init_discounted_utility *
                                      n_years)) * np.exp(-0.02005033585350133)
            self.welfare = welfare
            welfare_objective = np.asarray(
                [self.alpha * self.gamma * init_discounted_utility * n_years / welfare, ])
        else:
            pass
        return welfare_objective

    def compute_negative_welfare_objective(self):
        """
        Compute welfare objective as - welfare / init_discounted_utility * n_years
        """
        n_years = len(self.years_range)

        init_discounted_utility = self.init_discounted_utility
        welfare = self.utility_df['welfare'][self.year_end]

        self.welfare = welfare
        welfare_objective = np.asarray(
            [ - welfare / (init_discounted_utility * n_years)])
        return welfare_objective

    def compute_min_utility_objective(self):
        """
        Objective function: inputs : alpha, gamma and discounted_utility_ref
        """

        n_years = len(self.years_range)

        init_discounted_utility = self.init_discounted_utility
        min_utility = min(list(self.utility_df['discounted_utility'].values))
        # To avoid pb during convergence
        if min_utility / init_discounted_utility < 0.01:
            min_utility = 0.01 + \
                np.exp(min_utility / init_discounted_utility) * \
                np.exp(-0.02005033585350133)
        min_utility_objective = np.asarray(
            [self.alpha * (1 - self.gamma) * self.discounted_utility_ref / min_utility, ])
        return min_utility_objective

    ######### GRADIENTS ########

    def compute_gradient(self):

        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)

        # compute gradient
        d_period_utility_d_pc_consumption = np.zeros((nb_years, nb_years))
        d_discounted_utility_d_pc_consumption = np.zeros((nb_years, nb_years))
        d_discounted_utility_d_population = np.zeros((nb_years, nb_years))
        d_welfare_d_pc_consumption = np.zeros((nb_years, nb_years))
        d_welfare_d_population = np.zeros((nb_years, nb_years))

        for i in range(nb_years):
            pc_consumption = self.economics_df.at[years[i], GlossaryCore.PerCapitaConsumption]
            population = self.population_df.at[years[i], GlossaryCore.PopulationValue]
            u_discount_rate = self.utility_df.at[years[i], 'u_discount_rate']
            period_utility_pc = self.utility_df.at[years[i],
                                                   'period_utility_pc']
            energy_price = self.energy_mean_price.at[years[i], 'energy_price']
            period_utility_i = (
                pc_consumption**(1 - self.conso_elasticity) - 1) / (1 - self.conso_elasticity) - 1

#         #period_utility = (
#             pc_consumption**(1 - self.conso_elasticity) - 1) / (1 - self.conso_elasticity) - 1
#         adjusted_period_utility = period_utility * energy_price_ref/energy_price
            d_period_utility_d_pc_consumption[i,
                                              i] = pc_consumption ** (- self.conso_elasticity) *\
                self.energy_price_ref / energy_price

            if period_utility_i < self.min_period_utility:
                d_period_utility_d_pc_consumption[i, i] = d_period_utility_d_pc_consumption[i, i] * self.min_period_utility / 10. * (np.exp(
                    period_utility_i / self.min_period_utility) * np.exp(-1)) / self.min_period_utility
            d_discounted_utility_d_pc_consumption[i, i] = d_period_utility_d_pc_consumption[i, i] * \
                u_discount_rate * population

            d_discounted_utility_d_population[i, i] = period_utility_pc * u_discount_rate

            d_welfare_d_pc_consumption[nb_years - 1,
                                       i] = d_discounted_utility_d_pc_consumption[i, i]

            d_welfare_d_population[nb_years - 1, i] = d_discounted_utility_d_population[i, i]

        return d_period_utility_d_pc_consumption, d_discounted_utility_d_pc_consumption, d_discounted_utility_d_population,\
            d_welfare_d_pc_consumption, d_welfare_d_population

    def compute_gradient_energy_mean_price(self):
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)
        d_period_utility_d_energy_price = np.zeros((nb_years, nb_years))
        d_discounted_utility_d_energy_price = np.zeros((nb_years, nb_years))
        d_welfare_d_energy_price = np.zeros((nb_years, nb_years))

        pc_consumption = self.economics_df[GlossaryCore.PerCapitaConsumption].values
        population = self.population_df[GlossaryCore.PopulationValue].values
        u_discount_rate = self.utility_df['u_discount_rate'].values
        energy_price = self.energy_mean_price['energy_price'].values

        d_period_utility_d_energy_price = - 1.0 * \
            np.identity(nb_years) * \
            self.utility_df['period_utility_pc'].values / energy_price

        d_discounted_utility_d_energy_price = d_period_utility_d_energy_price * \
            u_discount_rate * population

        d_welfare_d_energy_price[nb_years - 1,
                                 ] = d_discounted_utility_d_energy_price.diagonal()
#             d_welfare_d_energy_price[nb_years - 1,
# 0] += d_discounted_utility_d_energy_price[i, 0]

        return d_period_utility_d_energy_price, d_discounted_utility_d_energy_price, d_welfare_d_energy_price

    def compute_gradient_objective(self):
        """
        if obj_option = 'last_utility': .alpha*init_utility/last_utility . with utility = period_utility_pc
        if obj_option = 'welfare :  alpha*init_discounted_utility*n_years/welfare
            if welfare < 1 : : alpha*initdiscounted_utility * n_years/ 
            (0.01+ np.exp(welfare/init_discounted_utility*n_years)*np.exp(-0.02005033585350133)) 
        """
        years = self.years_range
        period_utility_pc_0 = self.init_period_utility_pc
        period_utility_pc_end = self.utility_df.at[self.year_end,
                                                   'period_utility_pc']
        init_discounted_utility = self.init_discounted_utility

        n_years = len(years)

        d_obj_d_period_utility_pc = np.zeros(len(years))
        d_obj_d_welfare = np.zeros(len(years))

        if self.obj_option == 'last_utility':
            d_obj_d_period_utility_pc[-1] = -1.0 * self.alpha * \
                period_utility_pc_0 / (period_utility_pc_end)**2

        elif self.obj_option == 'welfare':
            welfare = self.utility_df['welfare'][self.year_end]
            if welfare / (init_discounted_utility * n_years) < 0.01:
                mask = np.append(np.zeros(len(years) - 1), np.array(1))
                f_prime = (1 / (init_discounted_utility * n_years)) * np.exp(welfare /
                                                                             (init_discounted_utility * n_years)) * np.exp(-0.02005033585350133)
                f_squared = (0.01 + np.exp(welfare / (init_discounted_utility *
                                                      n_years)) * np.exp(-0.02005033585350133))**2
                d_obj_d_welfare = mask * self.alpha * self.gamma *\
                    init_discounted_utility * n_years * (-f_prime / f_squared)
#                 mask = np.insert(np.zeros(len(years) - 1), 0, 1)
#                 u_prime_v = 0.01 + \
#                     np.exp(welfare / (init_discounted_utility *
#                                       n_years)) * np.exp(-0.02005033585350133)
#                 v_squared = (0.01 + np.exp(welfare / (init_discounted_utility *
#                                                       n_years)) * np.exp(-0.02005033585350133))**2
#                 v_prime_u = init_discounted_utility * np.exp(-0.02005033585350133) * (-welfare / (init_discounted_utility**2 * n_years)) *\
#                     np.exp(welfare / (init_discounted_utility * n_years))
#                 d_obj_d_discounted_utility = mask * self.alpha * \
#                     n_years * (u_prime_v - v_prime_u) / v_squared
            else:
                mask = np.append(np.zeros(len(years) - 1), np.array(1))
                d_obj_d_welfare = -1.0 * mask * self.alpha * self.gamma *\
                    init_discounted_utility * n_years / welfare**2

        else:
            pass

        return d_obj_d_welfare, d_obj_d_period_utility_pc

    def compute_gradient_negative_objective(self):
        """

        welfare = welfare / init_discounted_utility*n_years

        """
        years = self.years_range
        period_utility_pc_0 = self.init_period_utility_pc
        period_utility_pc_end = self.utility_df.at[self.year_end,
                                                   'period_utility_pc']
        init_discounted_utility = self.init_discounted_utility

        n_years = len(years)

        d_obj_d_period_utility_pc = np.zeros(len(years))
        d_obj_d_welfare = np.zeros(len(years))

        mask = np.append(np.zeros(len(years) - 1), np.array(1))
        d_obj_d_welfare = -1.0 * mask /  (init_discounted_utility * n_years)



        return d_obj_d_welfare, d_obj_d_period_utility_pc

    def compute_gradient_min_utility_objective(self):
        """

        """
        years = self.years_range
        init_discounted_utility = self.init_discounted_utility

        d_obj_d_period_utility_pc = np.zeros(len(years))
        d_obj_d_discounted_utility = np.zeros(len(years))

        min_utility = min(list(self.utility_df['discounted_utility'].values))
        d_min_utility_d_discounted_utility = np.asarray([1.0 if (val == min_utility) else 0. for val in list(
            self.utility_df['discounted_utility'].values)], dtype=float)
        if min_utility / init_discounted_utility < 0.01:

            f_prime = d_min_utility_d_discounted_utility * (1 / init_discounted_utility) * np.exp(min_utility /
                                                                                                  init_discounted_utility) * np.exp(-0.02005033585350133)
            f_squared = (0.01 + np.exp(min_utility / init_discounted_utility)
                         * np.exp(-0.02005033585350133))**2
            d_obj_d_discounted_utility = self.alpha * (1 - self.gamma) *\
                self.discounted_utility_ref * (-f_prime / f_squared)
        else:
            d_obj_d_discounted_utility = -1.0 * d_min_utility_d_discounted_utility * self.alpha * (1 - self.gamma) * \
                self.discounted_utility_ref / min_utility**2

        return d_obj_d_discounted_utility, d_obj_d_period_utility_pc

    def compute(self, economics_df, energy_mean_price, population_df):
        ''' pyworld3 execution
        '''
        self.economics_df = economics_df
        self.economics_df.index = self.economics_df[GlossaryCore.Years].values
        self.energy_mean_price = pd.DataFrame({GlossaryCore.Years: energy_mean_price[GlossaryCore.Years].values,
                                               'energy_price': energy_mean_price['energy_price'].values})
        self.energy_mean_price.index = self.energy_mean_price[GlossaryCore.Years].values
        self.energy_price_ref = self.initial_raw_energy_price
        self.population_df = population_df
        self.population_df.index = self.population_df[GlossaryCore.Years].values
        for year in self.years_range:
            self.compute__u_discount_rate(year)
            self.compute_period_utility(year)
            self.compute_discounted_utility(year)
        self.compute_welfare()

        return self.utility_df
