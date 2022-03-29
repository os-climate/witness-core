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
from numpy import fill_diagonal


class MacroEconomics():
    '''
    Economic model that compute the evolution of capital, consumption, output...
    '''
    PC_CONSUMPTION_CONSTRAINT = 'pc_consumption_constraint'

    def __init__(self, param):
        '''
        Constructor
        '''
        self.param = param
        self.inputs = None
        self.economics_df = None
        self.set_data()
        self.create_dataframe()

    def set_data(self):
        self.year_start = self.param['year_start']
        self.year_end = self.param['year_end']
        self.time_step = self.param['time_step']

        self.productivity_start = self.param['productivity_start']
        self.init_gross_output = self.param['init_gross_output']
        self.capital_start = self.param['capital_start']
        self.population_df = self.param['population_df']
#         self.population_df.index = self.population_df['years'].values
        self.productivity_gr_start = self.param['productivity_gr_start']
        self.decline_rate_tfp = self.param['decline_rate_tfp']
        self.depreciation_capital = self.param['depreciation_capital']
        self.init_rate_time_pref = self.param['init_rate_time_pref']
        self.conso_elasticity = self.param['conso_elasticity']
        self.lo_capital = self.param['lo_capital']
        self.lo_conso = self.param['lo_conso']
        self.lo_per_capita_conso = self.param['lo_per_capita_conso']
        self.hi_per_capita_conso = self.param['hi_per_capita_conso']
        self.ref_pc_consumption_constraint = self.param['ref_pc_consumption_constraint']
        self.nb_per = round(
            (self.param['year_end'] -
             self.param['year_start']) /
            self.param['time_step'] +
            1)
        self.years_range = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.frac_damage_prod = self.param['frac_damage_prod']
        self.damage_to_productivity = self.param['damage_to_productivity']
        self.init_energy_productivity_gr = self.param['init_energy_productivity_gr']
        self.init_energy_productivity = self.param['init_energy_productivity']
        self.init_output_growth = self.param['init_output_growth']
        self.decline_rate_energy_productivity = self.param['decline_rate_energy_productivity']
        # Param of output function
        self.output_k_exponent = self.param['output_k_exponent']
        self.output_pop_exponent = self.param['output_pop_exponent']
        self.output_energy_exponent = self.param['output_energy_exponent']
        self.output_exponent = self.param['output_exponent']
        self.output_pop_share = self.param['output_pop_share']
        self.output_energy_share = self.param['output_energy_share']
        self.pop_factor = self.param['pop_factor']
        self.energy_factor = self.param['energy_factor']
        self.co2_emissions_Gt = self.param['co2_emissions_Gt']
        self.co2_taxes = self.param['CO2_taxes']
        self.co2_tax_efficiency = self.param['CO2_tax_efficiency']
        self.scaling_factor_energy_investment = self.param['scaling_factor_energy_investment']

        if self.co2_tax_efficiency is not None:
            if 'years' in self.co2_tax_efficiency:
                self.co2_tax_efficiency.index = self.co2_tax_efficiency['years']
            else:
                raise Exception(
                    'Miss a column years in CO2 tax efficiency to set index of the dataframe')

        self.co2_invest_limit = self.param['co2_invest_limit']

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        default_index = np.arange(
            self.year_start, self.year_end + 1, self.time_step)
        param = self.param
        economics_df = pd.DataFrame(
            index=default_index,
            columns=['years',
                     'gross_output',
                     'output_net_of_d',
                     'net_output',
                     'productivity',
                     'productivity_gr',
                     'energy_productivity_gr',
                     'energy_productivity',
                     'consumption',
                     'pc_consumption',
                     'capital',
                     'investment',
                     'interest_rate',
                     'energy_investment',
                     'energy_investment_wo_tax',
                     'energy_investment_from_tax',
                     'output_growth'])

        for key in economics_df.keys():
            economics_df[key] = 0
        economics_df['years'] = self.years_range
        economics_df.loc[param['year_start'],
                         'gross_output'] = self.init_gross_output
        economics_df.loc[param['year_start'],
                         'energy_productivity'] = self.init_energy_productivity
        economics_df.loc[param['year_start'],
                         'energy_productivity_gr'] = self.init_energy_productivity_gr
        economics_df.loc[param['year_start'], 'capital'] = self.capital_start
        economics_df.loc[param['year_start'],
                         'productivity'] = self.productivity_start
        economics_df.loc[param['year_start'],
                         'productivity_gr'] = self.productivity_gr_start
        economics_df.loc[param['year_start'],
                         'output_growth'] = self.init_output_growth
#         economics_df['saving_rate'] = self.saving_rate
        self.economics_df = economics_df
        self.economics_df = self.economics_df.replace(
            [np.inf, -np.inf], np.nan)

        self.energy_investment_wo_renewable = pd.DataFrame(
            index=default_index,
            columns=['years',
                     'energy_investment_wo_renewable'])

        energy_investment = pd.DataFrame(
            index=default_index,
            columns=['years',
                     'energy_investment'])

        for key in energy_investment.keys():
            energy_investment[key] = 0
        energy_investment['years'] = self.years_range
        self.energy_investment = energy_investment
        self.energy_investment = self.energy_investment.replace(
            [np.inf, -np.inf], np.nan)

        return economics_df.fillna(0.0), energy_investment.fillna(0.0),

    def compute_productivity_growthrate(self, year):
        '''
        A_g, Growth rate of total factor productivity.
        Returns:
            :returns: A_g(0) * exp(-Δ_a * (t-1))
        '''
        t = ((year - self.year_start) / self.time_step) + 1
        productivity_gr = self.productivity_gr_start * \
            np.exp(-self.decline_rate_tfp * self.time_step * (t - 1))
        self.economics_df.loc[year, 'productivity_gr'] = productivity_gr
        return productivity_gr

    def compute_productivity(self, year):
        '''
        productivity
        if damage_to_productivity= True add damage to the the productivity
        if  not: productivity evolves independently from other variables (except productivity growthrate)
        '''
        damage_to_productivity = self.damage_to_productivity
        p_productivity = self.economics_df.at[year -
                                              self.time_step, 'productivity']
        p_productivity_gr = self.economics_df.at[year -
                                                 self.time_step, 'productivity_gr']
        damefrac = self.damefrac.at[year, 'damage_frac_output']
        if damage_to_productivity == True:
            #damage = 1-damefrac
            productivity = ((1 - self.frac_damage_prod * damefrac) *
                            (p_productivity / (1 - (p_productivity_gr / (5 / self.time_step)))))
        else:
            productivity = (p_productivity /
                            (1 - (p_productivity_gr / (5 / self.time_step))))
        # we divide the productivity growth rate by 5/time_step because of change in time_step (as
        # advised in Traeger, 2013)
        self.economics_df.loc[year, 'productivity'] = productivity
        return productivity

    def compute_energy_productivity_growthrate(self, year):
        '''
        Growth rate of energy productivity. 
        Same as productivity 
        Returns:
            :returns: Energy_productivity_gr(0) * exp(-Δ_a * (t-1))
        '''
        t = ((year - self.year_start) / self.time_step) + 1
        energy_productivity_gr = self.init_energy_productivity_gr * \
            np.exp(-self.decline_rate_energy_productivity *
                   self.time_step * (t - 1))
        self.economics_df.loc[year,
                              'energy_productivity_gr'] = energy_productivity_gr
        return energy_productivity_gr

    def compute_energy_productivity(self, year):
        '''
        Energy productivity.
        if damage to productivity = False:
        Energy productivity  evolves independently from other variables (except Energy productivity  growthrate)
        '''
        p_productivity = self.economics_df.at[year -
                                              self.time_step, 'energy_productivity']
        p_productivity_gr = self.economics_df.at[year -
                                                 self.time_step, 'energy_productivity_gr']
        damefrac = self.damefrac.at[year, 'damage_frac_output']

        if self.damage_to_productivity == True:
            #damage = 1-damefrac
            productivity = ((1 - self.frac_damage_prod * damefrac) *
                            (p_productivity / (1 - p_productivity_gr)))
        else:
            productivity = (p_productivity /
                            (1 - p_productivity_gr))
        # we divide the productivity growth rate by 5/time_step because of change in time_step (as
        # advised in Traeger, 2013)
        self.economics_df.loc[year, 'energy_productivity'] = productivity
        return productivity

    def compute_capital(self, year):
        """
        K(t+1), Capital for next time period, trillions $USD
        Args:
            :param capital: capital
            :param depreciation: depreciation rate
            :param investment: investment
        """
        if year == self.year_end:
            pass
        else:
            investment = self.economics_df.at[year, 'investment']
            capital = self.economics_df.at[year, 'capital']
            capital_a = capital * \
                (1 - self.depreciation_capital) ** self.time_step + \
                self.time_step * investment
            # Lower bound for capital
            self.economics_df.loc[year + self.time_step,
                                  'capital'] = max(capital_a, self.lo_capital)
            return capital_a

    def compute_investment(self, year):
        """
        I(t), Investment, trillions $USD

        """
#         saving_rate = self.saving_rate[year]
        net_output = self.economics_df.at[year, 'net_output']
#         investment = saving_rate * net_output
        energy_investment = self.economics_df.at[year,
                                                 'energy_investment']
        non_energy_investment = self.share_n_energy_investment[year] * net_output
        investment = energy_investment + non_energy_investment
        self.economics_df.loc[year, 'investment'] = investment
        return investment

    def compute_energy_investment(self, year):
        """
        energy_investment(t), trillions $USD (including renewable investments)
        Share of the total output 

        """
        net_output = self.economics_df.at[year, 'net_output']
        energy_investment_wo_tax = self.share_energy_investment[year] * net_output
        self.co2_emissions_Gt['Total CO2 emissions'].clip(
            lower=0.0, inplace=True)

        # store invests without renewable energy
        em_wo_ren = self.energy_investment_wo_renewable
        em_wo_ren.loc[year,
                      'energy_investment_wo_renewable'] = energy_investment_wo_tax * 1e3

        ren_investments = self.compute_energy_renewable_investment(
            year, energy_investment_wo_tax)
        energy_investment = energy_investment_wo_tax + ren_investments
        self.economics_df.loc[year,
                              ['energy_investment', 'energy_investment_wo_tax', 'energy_investment_from_tax']] = [energy_investment, energy_investment_wo_tax, ren_investments]
        self.energy_investment.loc[year,
                                   'energy_investment'] = energy_investment * 1e3 / self.scaling_factor_energy_investment  # Invest from T$ to G$ coupling variable

        return energy_investment

    def compute_energy_renewable_investment(self, year, energy_investment_wo_tax):
        """
        computes energy investment for renewable part
        for a given year: returns net CO2 emissions * CO2 taxes * a efficiency factor
        """
        co2_invest_limit = self.co2_invest_limit
        emissions = self.co2_emissions_Gt.at[year,
                                             'Total CO2 emissions'] * 1e9  # t CO2
        co2_taxes = self.co2_taxes.at[year, 'CO2_tax']  # $/t
        co2_tax_eff = self.co2_tax_efficiency.at[year,
                                                 'CO2_tax_efficiency'] / 100.  # %

        ren_investments = emissions * co2_taxes * co2_tax_eff / 1e12  # T$

        # if emissions is zero the right gradient (positive) is not zero but the left gradient is zero
        # when complex step we add ren_invest with the complex step and it is
        # not good
        if ren_investments.real == 0.0:
            ren_investments = 0.0
        # Saturation of renewable invest at n * invest wo tax with n ->
        # co2_invest_limit entry parameter
        if ren_investments > co2_invest_limit * energy_investment_wo_tax and ren_investments != 0.0:
            ren_investments = co2_invest_limit * energy_investment_wo_tax / 10.0 * \
                (9.0 + np.exp(- co2_invest_limit *
                              energy_investment_wo_tax / ren_investments))

        return ren_investments

    def compute_gross_output(self, year):
        """
        Gross output calculation: GDP PPP in constant 2020 US$. 
        Custom formula 
        Inputs : economics_df. capital(year), population(year), productivity(year),
        energy_productivity(year), energy supply(year)
        Output: the computed value of output
        """

        beta = self.output_k_exponent
        alpha = self.output_pop_exponent
        gamma = self.output_energy_exponent
        theta = self.output_exponent
        b = self.output_energy_share
        c = self.output_pop_share
        pop_factor = self.pop_factor
        energy_factor = self.energy_factor

        capital = self.economics_df.at[year, 'capital']
        population = self.population_df.at[year, 'population']
        productivity = self.economics_df.at[year, 'productivity']
        energy = self.energy_production.at[year,
                                           'Total production']
        energy_productivity = self.economics_df.at[year,
                                                   'energy_productivity']

        economic_part = productivity * \
            (c * (pop_factor * population)**alpha + (1 - c) * (capital)**beta)

        # replace negative values by 0 to avoid numpy warning
        energy = max(energy, 0)
        economic_part = max(economic_part, 0)
        energy_part = energy_productivity * (energy_factor * energy) ** gamma
        energy_part = max(energy_part, 0)

        output = (b * economic_part + (1 - b) * energy_part) ** theta
        self.economics_df.loc[year, 'gross_output'] = output
        return output

    def compute_gross_output_zhazhou(self, year):
        small_a = 0.9
        alpha = 2.126
        beta = 2.78
        gamma = 2.041
        b = 0.8638
        capital = self.economics_df.at[year, 'capital'] / \
            self.economics_df.at[self.year_start, 'capital']
        population = self.population_df.at[year, 'population'] / \
            self.population_df.at[self.year_start, 'population']
        productivity = self.economics_df.at[year, 'productivity'] / \
            self.economics_df.at[self.year_start, 'productivity']
        energy = self.energy_production.at[year, 'Total production'] / \
            self.energy_production.at[self.year_start,
                                      'Total production']
        energy_augm_techno = self.economics_df.at[year, 'energy_augm_techno'] / \
            self.economics_df.at[self.year_start, 'energy_augm_techno']
        cobb_douglas = productivity * \
            (small_a * (capital) ** (-alpha) + (1 - small_a)
             * (population)**(- alpha))**(-(1 / alpha))
        output_rel = (b * cobb_douglas**(- beta) + (1 - b) * (energy_augm_techno *
                                                              (energy / population)) ** (- beta))**(- gamma / beta)
        output = output_rel * self.init_gross_output
        self.economics_df.loc[year, 'gross_output'] = output
        return output

    def compute_output_growth(self, year):
        """
        Compute the output growth between year t and year t-1 
        Output growth of the WITNESS model (computed from gross_output_ter)
        """
        if year == self.year_start:
            pass
        else:
            gross_output_ter = self.economics_df.at[year,
                                                    'gross_output']
            gross_output_ter_a = self.economics_df.at[year -
                                                      self.time_step, 'gross_output']
            gross_output_ter = max(1e-6, gross_output_ter)
            output_growth = ((gross_output_ter -
                              gross_output_ter_a) / gross_output_ter) / self.time_step
            self.economics_df.loc[year, 'output_growth'] = output_growth
            return output_growth

    def compute_output_net_of_damage(self, year):
        """
        Output net of damages, trillions USD
        """
        damage_to_productivity = self.damage_to_productivity
        damefrac = self.damefrac.at[year, 'damage_frac_output']
        gross_output_ter = self.economics_df.at[year,
                                                'gross_output']
#        if damage_to_productivity == True :
#            D = 1 - damefrac
#            damage_to_output = D/(1-self.frac_damage_prod*(1-D))
#            output_net_of_d = gross_output * damage_to_output
#            damtoprod = D/(1-self.frac_damage_prod*(1-D))
        if damage_to_productivity == True:
            damage = 1 - ((1 - damefrac) /
                          (1 - self.frac_damage_prod * damefrac))
            output_net_of_d = (1 - damage) * gross_output_ter
        else:
            output_net_of_d = gross_output_ter * (1 - damefrac)
        self.economics_df.loc[year, 'output_net_of_d'] = output_net_of_d
        return output_net_of_d

    def compute_net_output(self, year):
        """
        Net output, trillions USD
        net_output = output_net_of_d - energy_investment_cost
        We concluded it is better to remove the energy_invest_cost, in this case net_output = output_net_of_d
        """
#         abatecost = self.abatecost[year]
#         output_net_of_d = self.economics_df.at[year, 'output_net_of_d']
#         net_output = output_net_of_d - abatecost
        #energy_invest_cost = self.economics_df.at[year, 'energy_invest_cost']
        output_net_of_d = self.economics_df.at[year, 'output_net_of_d']
        net_output = output_net_of_d  # - energy_invest_cost
        self.economics_df.loc[year,
                              'net_output'] = net_output
        return net_output

    def compute_consumption(self, year):
        """Equation for consumption
        C, Consumption, trillions $USD
        Args:
            output: Economic output at t
            savings: Savings rate at t
        """
        net_output = self.economics_df.at[year, 'net_output']
        investment = self.economics_df.at[year, 'investment']
        consumption = net_output - investment
        # lower bound for conso
        self.economics_df.loc[year, 'consumption'] = max(
            consumption, self.lo_conso)
        return consumption

    def compute_consumption_pc(self, year):
        """Equation for consumption per capita
        c, Per capita consumption, thousands $USD
        """
        consumption = self.economics_df.at[year, 'consumption']
        population = self.population_df.at[year, 'population']
        consumption_pc = consumption / population * 1000
        # Lower bound for pc conso
        self.economics_df.loc[year, 'pc_consumption'] = max(
            consumption_pc, self.lo_per_capita_conso)
        return consumption_pc

    def compute_comsumption_pc_constraint(self):
        """Equation for consumption per capita constraint
        c, Per capita consumption constraint
        """
        pc_consumption = self.economics_df['pc_consumption'].values
        self.pc_consumption_constraint = (self.hi_per_capita_conso - pc_consumption) \
            / self.ref_pc_consumption_constraint

    def compute_interest_rate(self, year):
        """Equation for interest rate
        """
        if year == self.year_end:
            pass
        else:
            consumption = self.economics_df.at[year, 'consumption']
            consumption_a = self.economics_df.at[year +
                                                 self.time_step, 'consumption']
            interest_rate = (1 + self.init_rate_time_pref) * (consumption_a /
                                                              consumption)**(self.conso_elasticity / self.time_step) - 1
            self.economics_df.loc[year, 'interest_rate'] = interest_rate
            return interest_rate

    """-------------------Gradient functions-------------------"""

    def compute_dproductivity(self):
        """gradient for productivity for damage_df
        Args:
            output: gradient
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)
        p_productivity_gr = self.economics_df['productivity_gr'].values
        p_productivity = self.economics_df['productivity'].values

        # derivative matrix initialization
        d_productivity = np.zeros((nb_years, nb_years))
        if self.damage_to_productivity == True:

            # first line stays at zero since derivatives of initial values are
            # zero
            for i in range(1, nb_years):
                d_productivity[i, i] = (1 - self.frac_damage_prod * self.damefrac.at[years[i], 'damage_frac_output']) * \
                    d_productivity[i - 1, i] / (1 - (p_productivity_gr[i - 1] /
                                                     (5 / self.time_step))) -\
                    self.frac_damage_prod * \
                    p_productivity[i - 1] / \
                    (1 - (p_productivity_gr[i - 1] / (5 / self.time_step)))
                for j in range(1, i):
                    d_productivity[i, j] = (1 - self.frac_damage_prod * self.damefrac.at[years[i], 'damage_frac_output']) * \
                        d_productivity[i - 1, j] / \
                        (1 - (p_productivity_gr[i - 1] / (5 / self.time_step)))

        return d_productivity

    def compute_denergy_productivity(self):
        """gradient for energy productivity for damage_df
        Args:
            output: gradient
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)
        p_productivity_gr = self.economics_df['energy_productivity_gr'].values
        p_productivity = self.economics_df['energy_productivity'].values

        # derivative matrix initialization
        denergy_productivity = np.zeros((nb_years, nb_years))
        if self.damage_to_productivity == True:

            # first line stays at zero since derivatives of initial values are
            # zero
            for i in range(1, nb_years):
                denergy_productivity[i, i] = (1 - self.frac_damage_prod * self.damefrac.at[years[i], 'damage_frac_output']) * denergy_productivity[i - 1, i] / (1 - (p_productivity_gr[i - 1])) -\
                    self.frac_damage_prod * \
                    p_productivity[i - 1] / \
                    (1 - (p_productivity_gr[i - 1]))
                for j in range(1, i):
                    denergy_productivity[i, j] = (1 - self.frac_damage_prod * self.damefrac.at[years[i], 'damage_frac_output']) * \
                        denergy_productivity[i - 1, j] / \
                        (1 - (p_productivity_gr[i - 1]))

        return denergy_productivity

    def compute_dgross_output_damage(self, denergy_productivity, dproductivity):
        """gradient computation for gross_output, capital, investment, energy_investment, net output for damage_df
        Args:
            input: gradients of energy_productivity, productivity
            output: gradients of gross_output, capital, investment, energy_investment, net output
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)

        beta = self.output_k_exponent
        alpha = self.output_pop_exponent
        gamma = self.output_energy_exponent
        theta = self.output_exponent
        b = self.output_energy_share
        c = self.output_pop_share
        pop_factor = self.pop_factor
        energy_factor = self.energy_factor

        # derivative matrix initialization
        dgross_output = np.zeros((nb_years, nb_years))
        dcapital = np.zeros((nb_years, nb_years))
        dinvestment = np.zeros((nb_years, nb_years))
        denergy_investment = np.zeros((nb_years, nb_years))
        dnet_output = np.zeros((nb_years, nb_years))

        # first line stays at zero since derivatives of initial values are zero
        for i in range(0, nb_years):

            population = self.population_df.at[years[i], 'population']
            capital = self.economics_df.at[years[i], 'capital']
            energy = self.energy_production.at[years[i],
                                               'Total production']
            productivity = self.economics_df.at[years[i], 'productivity']
            energy_productivity = self.economics_df.at[years[i],
                                                       'energy_productivity']
            output = self.economics_df.at[years[i], 'gross_output']
            damefrac = self.damefrac.at[years[i], 'damage_frac_output']

            emissions = self.co2_emissions_Gt.at[years[i],
                                                 'Total CO2 emissions']
            co2_taxes = self.co2_taxes.at[years[i], 'CO2_tax']
            co2_tax_eff = self.co2_tax_efficiency.at[years[i],
                                                     'CO2_tax_efficiency']
            energy_investment_wo_tax = self.economics_df.at[years[i],
                                                            'energy_investment_wo_tax']
            net_output = self.economics_df.at[years[i], 'net_output']

            ren_investments = emissions * 1e9 * co2_taxes * co2_tax_eff / 100 / 1e12  # T$

            for j in range(0, i + 1):

                if i != 0:
                    energy = max(energy, 0)
                    economic_part = max(0, productivity *
                                        (c * (pop_factor * population)**alpha + (1 - c) * (capital)**beta))
                    energy_part = max(0, energy_productivity *
                                      (energy_factor * energy)**gamma)
                    if energy_part == 0:
                        denergy_part = 0
                    else:
                        denergy_part = denergy_productivity[i,
                                                            j] * (energy_factor * energy)**gamma
                    if economic_part == 0:
                        deconomic_part = 0
                    else:
                        deconomic_part = dproductivity[i, j] * (c * (pop_factor * population)**alpha + (1 - c) * (capital)**beta) +\
                            productivity * beta * \
                            (1 - c) * dcapital[i, j] * (capital)**(beta - 1)

                    dgross_output[i, j] = theta * (b * deconomic_part + (1 - b) * denergy_part) * (
                        b * economic_part + (1 - b) * energy_part)**(theta - 1)

                if i == j:
                    if self.damage_to_productivity == True:
                        dnet_output[i, j] = (self.frac_damage_prod - 1) / ((self.frac_damage_prod * damefrac - 1)**2) * output + \
                            (1 - damefrac) / (1 - self.frac_damage_prod *
                                              damefrac) * dgross_output[i, j]
                    else:
                        dnet_output[i, j] = - output + \
                            (1 - damefrac) * dgross_output[i, j]
                else:
                    if self.damage_to_productivity == True:
                        dnet_output[i, j] = (
                            1 - damefrac) / (1 - self.frac_damage_prod * damefrac) * dgross_output[i, j]
                    else:
                        dnet_output[i, j] = (
                            1 - damefrac) * dgross_output[i, j]

                denergy_investment[i, j] = self.share_energy_investment[years[i]
                                                                        ] * dnet_output[i, j]

                # Saturation of renewable invest at n * invest wo tax with n -> co2_invest_limit entry parameter
                # ren_investments = self.co2_invest_limit * energy_investment_wo_tax / 10.0 * \
                #(9.0 + np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments))
                if ren_investments > self.co2_invest_limit * energy_investment_wo_tax:

                    denergy_investment[i, j] += self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] * 9 / 10 \
                        + self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] / 10 * np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments) \
                        + self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10 * (-1) * self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] / ren_investments \
                        * np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments)

                dinvestment[i, j] = denergy_investment[i, j] + \
                    self.share_n_energy_investment[years[i]
                                                   ] * dnet_output[i, j]

                if i < nb_years - 1:

                    capital_after = self.economics_df.at[years[i + 1], 'capital']
                    if capital_after == self.lo_capital:
                        dcapital[i + 1, j] = 0
                    else:
                        dcapital[i + 1, j] = dcapital[i, j] * (
                            1 - self.depreciation_capital) ** self.time_step + self.time_step * dinvestment[i, j]

        return dgross_output, dinvestment, denergy_investment, dnet_output

    def compute_dgross_output_denergy_supply(self):
        """gradient computation for gross_output, capital, investment, energy_investment, net output for energy_production
        Args:
            input: none
            output: gradients of gross_output, capital, investment, energy_investment, net output
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)

        beta = self.output_k_exponent
        alpha = self.output_pop_exponent
        gamma = self.output_energy_exponent
        theta = self.output_exponent
        b = self.output_energy_share
        c = self.output_pop_share
        pop_factor = self.pop_factor
        energy_factor = self.energy_factor

        # derivative matrix initialization
        dgross_output = np.zeros((nb_years, nb_years))
        dcapital = np.zeros((nb_years, nb_years))
        dinvestment = np.zeros((nb_years, nb_years))
        denergy_investment = np.zeros((nb_years, nb_years))
        dnet_output = np.zeros((nb_years, nb_years))

        economic_part = np.maximum(0, self.economics_df['productivity'].values *
                                   (c * (pop_factor * self.population_df['population'].values) **
                                    alpha + (1 - c) * (self.economics_df['capital'].values)**beta))
        energy_part = np.maximum(0, self.economics_df['energy_productivity'].values *
                                 (energy_factor * np.maximum(0, self.energy_production['Total production'].values))**gamma)

        # first line stays at zero since derivatives of initial values are zero
        for i in range(1, nb_years):

            capital_i = self.economics_df.at[years[i], 'capital']
            energy_i = self.energy_production.at[years[i],
                                                 'Total production']
            productivity_i = self.economics_df.at[years[i],
                                                  'productivity']
            energy_productivity_i = self.economics_df.at[years[i],
                                                         'energy_productivity']
            damefrac_i = self.damefrac.at[years[i], 'damage_frac_output']

            emissions = self.co2_emissions_Gt.at[years[i],
                                                 'Total CO2 emissions']
            co2_taxes = self.co2_taxes.at[years[i], 'CO2_tax']
            co2_tax_eff = self.co2_tax_efficiency.at[years[i],
                                                     'CO2_tax_efficiency']
            energy_investment_wo_tax = self.economics_df.at[years[i],
                                                            'energy_investment_wo_tax']
            net_output = self.economics_df.at[years[i], 'net_output']

            ren_investments = emissions * 1e9 * co2_taxes * co2_tax_eff / 100 / 1e12  # T$

            for j in range(1, i + 1):

                if energy_part[i] == 0:
                    denergy_part_i = 0
                else:
                    denergy_part_i = energy_productivity_i * energy_factor * \
                        gamma * (energy_factor * energy_i)**(gamma - 1)

                if economic_part[i] == 0:
                    deconomic_part = 0
                else:
                    deconomic_part = dcapital[i, j] * beta * productivity_i * \
                        (1 - c) * (capital_i)**(beta - 1)

                if i == j:
                    dgross_output[i, j] = theta * (b * deconomic_part + (1 - b) * denergy_part_i) * (
                        b * economic_part[i] + (1 - b) * energy_part[i])**(theta - 1)
                else:
                    dgross_output[i, j] = theta * (b * deconomic_part) * (
                        b * economic_part[i] + (1 - b) * energy_part[i])**(theta - 1)

                if self.damage_to_productivity == True:
                    dnet_output[i, j] = (
                        1 - damefrac_i) / (1 - self.frac_damage_prod * damefrac_i) * dgross_output[i, j]
                else:
                    dnet_output[i, j] = (1 - damefrac_i) * dgross_output[i, j]

                denergy_investment[i, j] = self.share_energy_investment[years[i]
                                                                        ] * dnet_output[i, j]

                # Saturation of renewable invest at n * invest wo tax with n ->
                # co2_invest_limit entry parameter
                if ren_investments > self.co2_invest_limit * energy_investment_wo_tax:

                    denergy_investment[i, j] += self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] * 9 / 10 \
                        + self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] / 10 * np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments) \
                        + self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10 * (-1) * self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] / ren_investments \
                        * np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments)

                dinvestment[i, j] = denergy_investment[i, j] + \
                    self.share_n_energy_investment[years[i]
                                                   ] * dnet_output[i, j]

                if i < nb_years - 1:
                    capital_after = self.economics_df.at[years[i + 1], 'capital']
                    if capital_after == self.lo_capital:
                        dcapital[i + 1, j] = 0
                    else:
                        dcapital[i + 1, j] = dcapital[i, j] * (
                            1 - self.depreciation_capital) ** self.time_step + self.time_step * dinvestment[i, j]

        return dgross_output, dinvestment, denergy_investment, dnet_output

    def compute_dgross_output_dCO2_emission_gt(self):
        """gradient computation for gross_output, capital, investment, energy_investment, net output for energy_production
        Args:
            input: none
            output: gradients of gross_output, capital, investment, energy_investment, net output
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)

        beta = self.output_k_exponent
        alpha = self.output_pop_exponent
        gamma = self.output_energy_exponent
        theta = self.output_exponent
        b = self.output_energy_share
        c = self.output_pop_share
        pop_factor = self.pop_factor
        energy_factor = self.energy_factor

        # derivative matrix initialization
        dgross_output = np.zeros((nb_years, nb_years))
        dcapital = np.zeros((nb_years, nb_years))
        dinvestment = np.zeros((nb_years, nb_years))
        denergy_investment = np.zeros((nb_years, nb_years))
        dnet_output = np.zeros((nb_years, nb_years))

        economic_part = np.maximum(0, self.economics_df['productivity'].values *
                                   (c * (pop_factor * self.population_df['population'].values) **
                                    alpha + (1 - c) * (self.economics_df['capital'].values)**beta))
        energy_part = np.maximum(0, self.economics_df['energy_productivity'].values *
                                 (energy_factor * np.maximum(0, self.energy_production['Total production'].values))**gamma)

        # first line stays at zero since derivatives of initial values are zero
        for i in range(0, nb_years):

            capital_i = self.economics_df.at[years[i], 'capital']
            productivity_i = self.economics_df.at[years[i],
                                                  'productivity']
            damefrac_i = self.damefrac.at[years[i], 'damage_frac_output']

            emissions = self.co2_emissions_Gt.at[years[i],
                                                 'Total CO2 emissions']
            co2_taxes = self.co2_taxes.at[years[i], 'CO2_tax']
            co2_tax_eff = self.co2_tax_efficiency.at[years[i],
                                                     'CO2_tax_efficiency']
            energy_investment_wo_tax = self.economics_df.at[years[i],
                                                            'energy_investment_wo_tax']
            net_output = self.economics_df.at[years[i], 'net_output']

            ren_investments = emissions * 1e9 * co2_taxes * co2_tax_eff / 100 / 1e12  # T$

            for j in range(0, i + 1):

                if economic_part[i] == 0:
                    deconomic_part = 0
                else:
                    deconomic_part = dcapital[i, j] * beta * productivity_i * \
                        (1 - c) * (capital_i)**(beta - 1)

                dgross_output[i, j] = theta * (b * deconomic_part) * (
                    b * economic_part[i] + (1 - b) * energy_part[i])**(theta - 1)

                if self.damage_to_productivity == True:
                    dnet_output[i, j] = (
                        1 - damefrac_i) / (1 - self.frac_damage_prod * damefrac_i) * dgross_output[i, j]
                else:
                    dnet_output[i, j] = (1 - damefrac_i) * dgross_output[i, j]

                if i == j:
                    denergy_investment[i, j] = self.share_energy_investment[years[i]
                                                                            ] * dnet_output[i, j] + np.sign(emissions) * co2_taxes * co2_tax_eff * 1e9 / 1e12  # T$

                    # Saturation of renewable invest at n * invest wo tax with
                    # n -> co2_invest_limit entry parameter
                    if ren_investments > self.co2_invest_limit * energy_investment_wo_tax:
                        # Base function:
                        # ren_investments = self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10.0 * \
                        #    (9.0 + np.exp(- self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / ren_investments))
                        # Derivative:
                        #dren_investments = d(self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output * 9 / 10.0)
                        #  + d(self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10.0
                        #  * np.exp(- self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / ren_investments))
                        # So d(u*v) = u'.v + u.v' with:
                        #  u = self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10.0
                        #  v = np.exp(- self.co2_invest_limit * self.share_energy_investment[year] * net_output / ren_investments)
                        # With v'= d(- self.co2_invest_limit * self.share_energy_investment[year] * net_output / ren_investments) * np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments)
                        # So d(w/x) = w'.x - w.x' / (x*x) with:
                        #  w = - self.co2_invest_limit * self.share_energy_investment[year] * net_output
                        #  x = ren_investments = emissions * co2_taxes * co2_tax_eff / 1e12 * 1e9
                        denergy_investment[i, j] = self.share_energy_investment[years[i]] * dnet_output[i, j] \
                            + self.co2_invest_limit * dnet_output[i, j] * self.share_energy_investment[years[i]] * 9 / 10 \
                            + self.co2_invest_limit * dnet_output[i, j] * self.share_energy_investment[years[i]] / 10 * np.exp(- self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / ren_investments) \
                            + self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10 * (- self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] * ren_investments
                                                                                                                  + self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output * co2_taxes * co2_tax_eff * 1e9 / 1e12) / (ren_investments * ren_investments) * np.exp(- self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / ren_investments)

                else:
                    denergy_investment[i, j] = self.share_energy_investment[years[i]
                                                                            ] * dnet_output[i, j]

                    if ren_investments > self.co2_invest_limit * energy_investment_wo_tax:
                        denergy_investment[i, j] += self.co2_invest_limit * dnet_output[i, j] * self.share_energy_investment[years[i]] * 9 / 10 \
                            + self.co2_invest_limit * dnet_output[i, j] * self.share_energy_investment[years[i]] / 10 * np.exp(- self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / ren_investments) \
                            + self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10 * (-1) * self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] * ren_investments \
                            / (ren_investments * ren_investments) * np.exp(- self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / ren_investments)

                dinvestment[i, j] = denergy_investment[i, j] + \
                    self.share_n_energy_investment[years[i]
                                                   ] * dnet_output[i, j]

                if i < nb_years - 1:
                    capital_after = self.economics_df.at[years[i + 1], 'capital']
                    if capital_after == self.lo_capital:
                        dcapital[i + 1, j] = 0
                    else:
                        dcapital[i + 1, j] = dcapital[i, j] * (
                            1 - self.depreciation_capital) ** self.time_step + self.time_step * dinvestment[i, j]

        return dgross_output, dinvestment, denergy_investment, dnet_output

    def compute_dgross_output_dCO2_taxes(self):
        """gradient computation for gross_output, capital, investment, energy_investment, net output for energy_production
        Args:
            input: none
            output: gradients of gross_output, capital, investment, energy_investment, net output
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)

        beta = self.output_k_exponent
        alpha = self.output_pop_exponent
        gamma = self.output_energy_exponent
        theta = self.output_exponent
        b = self.output_energy_share
        c = self.output_pop_share
        pop_factor = self.pop_factor
        energy_factor = self.energy_factor

        # derivative matrix initialization
        dgross_output = np.zeros((nb_years, nb_years))
        dcapital = np.zeros((nb_years, nb_years))
        dinvestment = np.zeros((nb_years, nb_years))
        denergy_investment = np.zeros((nb_years, nb_years))
        dnet_output = np.zeros((nb_years, nb_years))

        economic_part = np.maximum(0, self.economics_df['productivity'].values *
                                   (c * (pop_factor * self.population_df['population'].values) **
                                    alpha + (1 - c) * (self.economics_df['capital'].values)**beta))
        energy_part = np.maximum(0, self.economics_df['energy_productivity'].values *
                                 (energy_factor * np.maximum(0, self.energy_production['Total production'].values))**gamma)

        # first line stays at zero since derivatives of initial values are zero
        for i in range(0, nb_years):

            capital_i = self.economics_df.at[years[i], 'capital']
            productivity_i = self.economics_df.at[years[i],
                                                  'productivity']
            damefrac_i = self.damefrac.at[years[i], 'damage_frac_output']

            emissions = self.co2_emissions_Gt.at[years[i],
                                                 'Total CO2 emissions']
            co2_taxes = self.co2_taxes.at[years[i], 'CO2_tax']
            co2_tax_eff = self.co2_tax_efficiency.at[years[i],
                                                     'CO2_tax_efficiency']
            energy_investment_wo_tax = self.economics_df.at[years[i],
                                                            'energy_investment_wo_tax']
            net_output = self.economics_df.at[years[i], 'net_output']

            ren_investments = emissions * 1e9 * co2_taxes * co2_tax_eff / 100 / 1e12  # T$

            # %
            co2_tax_eff = self.co2_tax_efficiency.at[years[i],
                                                     'CO2_tax_efficiency']

            for j in range(0, i + 1):
                if economic_part[i] == 0:
                    deconomic_part = 0
                else:
                    deconomic_part = dcapital[i, j] * beta * productivity_i * \
                        (1 - c) * (capital_i)**(beta - 1)

                dgross_output[i, j] = theta * (b * deconomic_part) * (
                    b * economic_part[i] + (1 - b) * energy_part[i])**(theta - 1)

                if self.damage_to_productivity == True:
                    dnet_output[i, j] = (
                        1 - damefrac_i) / (1 - self.frac_damage_prod * damefrac_i) * dgross_output[i, j]
                else:
                    dnet_output[i, j] = (1 - damefrac_i) * dgross_output[i, j]

                if i == j:
                    denergy_investment[i, j] = self.share_energy_investment[years[i]
                                                                            ] * dnet_output[i, j] + emissions * co2_tax_eff * 1e9 / 1e12  # T$

                    # Saturation of renewable invest at n * invest wo tax with
                    # n -> co2_invest_limit entry parameter
                    if ren_investments > self.co2_invest_limit * energy_investment_wo_tax:
                        # Base function:
                        # ren_investments = self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10.0 * \
                        #    (9.0 + np.exp(- self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / ren_investments))
                        # Derivative:
                        #dren_investments = d(self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output * 9 / 10.0)
                        #  + d(self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10.0
                        #  * np.exp(- self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / ren_investments))
                        # So d(u*v) = u'.v + u.v' with:
                        #  u = self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10.0
                        #  v = np.exp(- self.co2_invest_limit * self.share_energy_investment[year] * net_output / ren_investments)
                        # With v'= d(- self.co2_invest_limit * self.share_energy_investment[year] * net_output / ren_investments) * np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments)
                        # So d(w/x) = w'.x - w.x' / (x*x) with:
                        #  w = - self.co2_invest_limit * self.share_energy_investment[year] * net_output
                        #  x = ren_investments = emissions * co2_taxes * co2_tax_eff / 1e12 * 1e9
                        denergy_investment[i, j] = self.share_energy_investment[years[i]] * dnet_output[i, j] \
                            + self.co2_invest_limit * dnet_output[i, j] * self.share_energy_investment[years[i]] * 9 / 10 \
                            + self.co2_invest_limit * dnet_output[i, j] * self.share_energy_investment[years[i]] / 10 * np.exp(- self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / ren_investments) \
                            + self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10 * (- self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] * ren_investments
                                                                                                                  + self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output * emissions * co2_tax_eff * 1e9 / 1e12) / (ren_investments * ren_investments) * np.exp(- self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / ren_investments)

                else:
                    denergy_investment[i, j] = self.share_energy_investment[years[i]
                                                                            ] * dnet_output[i, j]

                    if ren_investments > self.co2_invest_limit * energy_investment_wo_tax:
                        denergy_investment[i, j] += self.co2_invest_limit * dnet_output[i, j] * self.share_energy_investment[years[i]] * 9 / 10 \
                            + self.co2_invest_limit * dnet_output[i, j] * self.share_energy_investment[years[i]] / 10 * np.exp(- self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / ren_investments) \
                            + self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10 * (-1) * self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] * ren_investments \
                            / (ren_investments * ren_investments) * np.exp(- self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / ren_investments)

                dinvestment[i, j] = denergy_investment[i, j] + \
                    self.share_n_energy_investment[years[i]
                                                   ] * dnet_output[i, j]

                if i < nb_years - 1:
                    capital_after = self.economics_df.at[years[i + 1], 'capital']
                    if capital_after == self.lo_capital:
                        dcapital[i + 1, j] = 0
                    else:
                        dcapital[i + 1, j] = dcapital[i, j] * (
                            1 - self.depreciation_capital) ** self.time_step + self.time_step * dinvestment[i, j]

        return dgross_output, dinvestment, denergy_investment, dnet_output

    def compute_dgross_output_dpopulation(self):
        """gradient computation for gross_output, capital, investment, energy_investment, net output for population
        Args:
            input: none
            output: gradients of gross_output, capital, investment, energy_investment, net output
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)

        beta = self.output_k_exponent
        alpha = self.output_pop_exponent
        gamma = self.output_energy_exponent
        theta = self.output_exponent
        b = self.output_energy_share
        c = self.output_pop_share
        pop_factor = self.pop_factor
        energy_factor = self.energy_factor

        # derivative matrix initialization
        dgross_output = np.zeros((nb_years, nb_years))
        dcapital = np.zeros((nb_years, nb_years))
        dinvestment = np.zeros((nb_years, nb_years))
        denergy_investment = np.zeros((nb_years, nb_years))
        dnet_output = np.zeros((nb_years, nb_years))

        economic_part = np.maximum(0, self.economics_df['productivity'].values *
                                   (c * (pop_factor * self.population_df['population'].values) **
                                    alpha + (1 - c) * (self.economics_df['capital'].values)**beta))
        energy_part = np.maximum(0, self.economics_df['energy_productivity'].values *
                                 (energy_factor * np.maximum(0, self.energy_production['Total production'].values))**gamma)

        # first line stays at zero since derivatives of initial values are zero
        for i in range(1, nb_years):

            capital_i = self.economics_df.at[years[i], 'capital']
            energy_i = self.energy_production.at[years[i],
                                                 'Total production']
            productivity_i = self.economics_df.at[years[i],
                                                  'productivity']
            energy_productivity_i = self.economics_df.at[years[i],
                                                         'energy_productivity']
            damefrac_i = self.damefrac.at[years[i], 'damage_frac_output']

            emissions = self.co2_emissions_Gt.at[years[i],
                                                 'Total CO2 emissions']
            co2_taxes = self.co2_taxes.at[years[i], 'CO2_tax']
            co2_tax_eff = self.co2_tax_efficiency.at[years[i],
                                                     'CO2_tax_efficiency']
            energy_investment_wo_tax = self.economics_df.at[years[i],
                                                            'energy_investment_wo_tax']
            net_output = self.economics_df.at[years[i], 'net_output']

            ren_investments = emissions * 1e9 * co2_taxes * co2_tax_eff / 100 / 1e12  # T$

            population = self.population_df.at[years[i], 'population']

            for j in range(1, i + 1):

                if economic_part[i] == 0:
                    deconomic_part = 0
                else:
                    if i == j:
                        deconomic_part = productivity_i * c * pop_factor * alpha * (pop_factor * population) ** (alpha - 1) + dcapital[i, j] * beta * productivity_i * \
                            (1 - c) * (capital_i) ** (beta - 1)
                    else:
                        deconomic_part = dcapital[i, j] * beta * \
                            productivity_i * (1 - c) * \
                            (capital_i) ** (beta - 1)

                dgross_output[i, j] = theta * (b * deconomic_part) * (
                    b * economic_part[i] + (1 - b) * energy_part[i]) ** (theta - 1)

                if self.damage_to_productivity == True:
                    dnet_output[i, j] = (
                        1 - damefrac_i) / (1 - self.frac_damage_prod * damefrac_i) * dgross_output[i, j]
                else:
                    dnet_output[i, j] = (1 - damefrac_i) * dgross_output[i, j]

                denergy_investment[i, j] = self.share_energy_investment[years[i]
                                                                        ] * dnet_output[i, j]

                # Saturation of renewable invest at n * invest wo tax with n ->
                # co2_invest_limit entry parameter
                if ren_investments > self.co2_invest_limit * energy_investment_wo_tax:

                    denergy_investment[i, j] += self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] * 9 / 10 \
                        + self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] / 10 * np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments) \
                        + self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10 * (-1) * self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] / ren_investments \
                        * np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments)

                dinvestment[i, j] = denergy_investment[i, j] + \
                    self.share_n_energy_investment[years[i]
                                                   ] * dnet_output[i, j]

                if i < nb_years - 1:
                    capital_after = self.economics_df.at[years[i + 1], 'capital']
                    if capital_after == self.lo_capital:
                        dcapital[i + 1, j] = 0
                    else:
                        dcapital[i + 1, j] = dcapital[i, j] * (
                            1 - self.depreciation_capital) ** self.time_step + self.time_step * dinvestment[i, j]

        return dgross_output, dinvestment, denergy_investment, dnet_output

    def compute_dtotal_investment_share_of_gdp(self):
        """gradient computation for gross_output, capital, investment, energy_investment, net output for total_investment_share_of_gdp
        Args:
            output: gradients of gross_output, capital, investment, energy_investment, net output
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)

        beta = self.output_k_exponent
        alpha = self.output_pop_exponent
        gamma = self.output_energy_exponent
        theta = self.output_exponent
        b = self.output_energy_share
        c = self.output_pop_share
        pop_factor = self.pop_factor
        energy_factor = self.energy_factor

        # derivative matrix initialization
        dgross_output = np.zeros((nb_years, nb_years))
        dcapital = np.zeros((nb_years, nb_years))
        dinvestment = np.zeros((nb_years, nb_years))
        denergy_investment = np.zeros((nb_years, nb_years))
        dnet_output = np.zeros((nb_years, nb_years))

        # first line stays at zero since derivatives of initial values are zero
        for i in range(0, nb_years):

            population = self.population_df.at[years[i], 'population']
            capital = self.economics_df.at[years[i], 'capital']
            energy = self.energy_production.at[years[i],
                                               'Total production']
            productivity = self.economics_df.at[years[i], 'productivity']
            energy_productivity = self.economics_df.at[years[i],
                                                       'energy_productivity']
            net_output = self.economics_df.at[years[i], 'net_output']
            damefrac = self.damefrac.at[years[i], 'damage_frac_output']

            emissions = self.co2_emissions_Gt.at[years[i],
                                                 'Total CO2 emissions']
            co2_taxes = self.co2_taxes.at[years[i], 'CO2_tax']
            co2_tax_eff = self.co2_tax_efficiency.at[years[i],
                                                     'CO2_tax_efficiency']
            energy_investment_wo_tax = self.economics_df.at[years[i],
                                                            'energy_investment_wo_tax']

            ren_investments = emissions * 1e9 * co2_taxes * co2_tax_eff / 100 / 1e12  # T$

            for j in range(0, i + 1):
                if i != 0:
                    energy = max(0, energy)
                    economic_part = max(0, productivity *
                                        (c * (pop_factor * population)**alpha + (1 - c) * (capital)**beta))
                    energy_part = max(0, energy_productivity *
                                      (energy_factor * energy)**gamma)
                    if economic_part == 0:
                        deconomic_part = 0
                    else:
                        deconomic_part = beta * \
                            dcapital[i, j] * productivity * \
                            ((1 - c) * (capital)**(beta - 1))

                    dgross_output[i, j] = theta * b * deconomic_part * \
                        (b * economic_part + (1 - b) * energy_part)**(theta - 1)

                    if self.damage_to_productivity == True:
                        dnet_output[i, j] = (
                            1 - damefrac) / (1 - self.frac_damage_prod * damefrac) * dgross_output[i, j]
                    else:
                        dnet_output[i, j] = (
                            1 - damefrac) * dgross_output[i, j]

                denergy_investment[i, j] = self.share_energy_investment[years[i]
                                                                        ] * dnet_output[i, j]

                # Saturation of renewable invest at n * invest wo tax with n ->
                # co2_invest_limit entry parameter
                if ren_investments > self.co2_invest_limit * energy_investment_wo_tax:

                    denergy_investment[i, j] += self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] * 9 / 10 \
                        + self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] / 10 * np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments) \
                        + self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10 * (-1) * self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] / ren_investments \
                        * np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments)

                if i == j:
                    dinvestment[i, j] = denergy_investment[i, j] + \
                        self.share_n_energy_investment[years[i]
                                                       ] * dnet_output[i, j] + net_output
                else:
                    dinvestment[i, j] = denergy_investment[i, j] + \
                        self.share_n_energy_investment[years[i]
                                                       ] * dnet_output[i, j]

                if i < nb_years - 1:
                    capital_after = self.economics_df.at[years[i + 1], 'capital']
                    if capital_after == self.lo_capital:
                        dcapital[i + 1, j] = 0
                    else:
                        dcapital[i + 1, j] = dcapital[i, j] * (
                            1 - self.depreciation_capital) ** self.time_step + self.time_step * dinvestment[i, j]

        return dgross_output, dinvestment, denergy_investment, dnet_output

    def compute_dshare_energy_investment(self):
        """gradient computation for gross_output, capital, investment, energy_investment, net output for energy_production
        Args:
            input: gradients of energy_productivity, productivity
            output: gradients of gross_output, capital, investment, energy_investment, net output
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)

        beta = self.output_k_exponent
        alpha = self.output_pop_exponent
        gamma = self.output_energy_exponent
        theta = self.output_exponent
        b = self.output_energy_share
        c = self.output_pop_share
        pop_factor = self.pop_factor
        energy_factor = self.energy_factor

        # derivative matrix initialization
        dgross_output = np.zeros((nb_years, nb_years))
        dcapital = np.zeros((nb_years, nb_years))
        dinvestment = np.zeros((nb_years, nb_years))
        denergy_investment = np.zeros((nb_years, nb_years))
        dnet_output = np.zeros((nb_years, nb_years))

        economic_part = np.maximum(0, self.economics_df['productivity'].values *
                                   (c * (pop_factor * self.population_df['population'].values) **
                                    alpha + (1 - c) * (self.economics_df['capital'].values)**beta))
        energy_part = np.maximum(0, self.economics_df['energy_productivity'].values *
                                 (energy_factor * np.maximum(0, self.energy_production['Total production'].values))**gamma)

        # first line stays at zero since derivatives of initial values are zero
        for i in range(0, nb_years):

            capital_i = self.economics_df.at[years[i], 'capital']
            productivity_i = self.economics_df.at[years[i],
                                                  'productivity']
            damefrac_i = self.damefrac.at[years[i], 'damage_frac_output']

            co2_tax_eff = self.co2_tax_efficiency.at[years[i],
                                                     'CO2_tax_efficiency']
            emissions = self.co2_emissions_Gt.at[years[i],
                                                 'Total CO2 emissions']
            co2_taxes = self.co2_taxes.at[years[i], 'CO2_tax']
            co2_tax_eff = self.co2_tax_efficiency.at[years[i],
                                                     'CO2_tax_efficiency']
            energy_investment_wo_tax = self.economics_df.at[years[i],
                                                            'energy_investment_wo_tax']
            net_output = self.economics_df.at[years[i], 'net_output']

            ren_investments = emissions * 1e9 * co2_taxes * co2_tax_eff / 100 / 1e12  # T$

            for j in range(0, i + 1):

                if economic_part[i] == 0:
                    deconomic_part = 0
                else:
                    deconomic_part = dcapital[i, j] * beta * productivity_i * \
                        (1 - c) * (capital_i)**(beta - 1)

                dgross_output[i, j] = theta * (b * deconomic_part) * (
                    b * economic_part[i] + (1 - b) * energy_part[i])**(theta - 1)

                if self.damage_to_productivity == True:
                    dnet_output[i, j] = (
                        1 - damefrac_i) / (1 - self.frac_damage_prod * damefrac_i) * dgross_output[i, j]
                else:
                    dnet_output[i, j] = (1 - damefrac_i) * dgross_output[i, j]

                if i == j:
                    denergy_investment[i, j] = net_output + \
                        self.share_energy_investment[years[i]
                                                     ] * dnet_output[i, j]

                    # Saturation of renewable invest at n * invest wo tax with
                    # n -> co2_invest_limit entry parameter
                    if ren_investments > self.co2_invest_limit * energy_investment_wo_tax:
                        # Base function:
                        # ren_investments = self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10.0 * \
                        #    (9.0 + np.exp(- self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / ren_investments))
                        # Derivative:
                        #dren_investments = d(self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output * 9 / 10.0)
                        #  + d(self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10.0
                        #  * np.exp(- self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / ren_investments))
                        # So d(u*v*w) = u'.v.w + u.v'.w + u.v.w' with:
                        #  u = self.co2_invest_limit * self.share_energy_investment[years[i]]
                        #  v = net_output / 10.0
                        #  w = np.exp(- self.co2_invest_limit * self.share_energy_investment[year] * net_output / ren_investments)
                        # With w'= d(- self.co2_invest_limit *
                        # self.share_energy_investment[year] * net_output /
                        # ren_investments) * np.exp(- self.co2_invest_limit *
                        # energy_investment_wo_tax / ren_investments)
                        denergy_investment[i, j] += (self.co2_invest_limit * net_output + self.co2_invest_limit * dnet_output[i, j] * self.share_energy_investment[years[i]]) * 9 / 10 \
                            + self.co2_invest_limit * net_output / 10 * np.exp(- self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / ren_investments) \
                            + self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] / 10 * np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments) \
                            + self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10 * (- self.co2_invest_limit * net_output - self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j]) / ren_investments \
                            * np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments)

                else:
                    # Same methodology but
                    # self.share_energy_investment[years[i]] is identity matrix
                    # so u' = 0
                    if ren_investments > self.co2_invest_limit * energy_investment_wo_tax:

                        denergy_investment[i, j] = self.share_energy_investment[years[i]] * dnet_output[i, j] + \
                            self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] * 9 / 10 \
                            + self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j] / 10 * np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments) \
                            + self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10 * (- self.co2_invest_limit * self.share_energy_investment[years[i]] * dnet_output[i, j]) / ren_investments \
                            * np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments)
                    else:
                        denergy_investment[i, j] = self.share_energy_investment[years[i]
                                                                                ] * dnet_output[i, j]

                if i == j:
                    dinvestment[i, j] = denergy_investment[i, j] + \
                        self.share_n_energy_investment[years[i]
                                                       ] * dnet_output[i, j] - net_output
                else:
                    dinvestment[i, j] = denergy_investment[i, j] + \
                        self.share_n_energy_investment[years[i]
                                                       ] * dnet_output[i, j]

                if i < nb_years - 1:
                    capital_after = self.economics_df.at[years[i + 1], 'capital']
                    if capital_after == self.lo_capital:
                        dcapital[i + 1, j] = 0
                    else:
                        dcapital[i + 1, j] = dcapital[i, j] * (
                            1 - self.depreciation_capital) ** self.time_step + self.time_step * dinvestment[i, j]

        return dgross_output, dinvestment, denergy_investment, dnet_output

    def compute_doutput_growth(self, dgross_output):
        """gradient computation for output_growth
        Args:
            input: gradients of gross_output
            output: gradients of output_growth
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)

        # derivative matrix initialization
        doutput_growth = np.zeros((nb_years, nb_years))

        # first line stays at zero since derivatives of initial values are zero
        for i in range(1, nb_years):
            gross_output_ter = self.economics_df.at[years[i], 'gross_output']
            gross_output_ter_a = self.economics_df.at[years[i] -
                                                      self.time_step, 'gross_output']
            for j in range(0, i + 1):
                doutput_growth[i, j] = ((dgross_output[i, j] - dgross_output[i - 1, j]) * gross_output_ter - (gross_output_ter -
                                                                                                              gross_output_ter_a) * dgross_output[i, j]) / (gross_output_ter**2) / self.time_step

        return doutput_growth

    def compute_dconsumption(self, dnet_output, dinvestment):
        """gradient computation for consumption
        Args:
            input: gradients of net_output, investment
            output: gradients of consumption
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)

        # derivative matrix initialization
        dconsumption = np.zeros((nb_years, nb_years))

        # first line stays at zero since derivatives of initial values are zero
        for i in range(0, nb_years):
            for j in range(0, i + 1):
                consumption = self.economics_df.at[years[i], 'consumption']
                if consumption == self.lo_conso:
                    pass
                else:
                    dconsumption[i, j] = dnet_output[i, j] - dinvestment[i, j]

        return dconsumption

    def compute_dconsumption_pc(self, dconsumption):
        """gradient computation for pc_consumption
        Args:
            input: gradients of consumption
            output: gradients of pc_consumption
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)

        # derivative matrix initialization
        dconsumption_pc = np.zeros((nb_years, nb_years))

        # first line stays at zero since derivatives of initial values are zero
        for i in range(0, nb_years):
            consumption_pc = self.economics_df.at[years[i], 'pc_consumption']
            if consumption_pc > self.lo_per_capita_conso:
                for j in range(0, i + 1):
                    dconsumption_pc[i, j] = dconsumption[i, j] / \
                        self.population_df.at[years[i], 'population'] * 1000

        return dconsumption_pc

    def compute_dconsumption_pc_dpopulation(self, dconsumption):
        """gradient computation for pc_consumption
        Args:
            input: gradients of consumption
            output: gradients of pc_consumption
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)

        # derivative matrix initialization
        dconsumption_pc = np.zeros((nb_years, nb_years))

        # first line stays at zero since derivatives of initial values are zero
        for i in range(0, nb_years):
            consumption_pc = self.economics_df.at[years[i], 'pc_consumption']
            consumption = self.economics_df.at[years[i], 'consumption']
            population = self.population_df.at[years[i], 'population']
            if consumption_pc > self.lo_per_capita_conso:
                for j in range(0, i + 1):
                    if i == j:
                        dconsumption_pc[i, j] = (
                            dconsumption[i, j] * population - consumption) / (population * population) * 1000
                    else:
                        dconsumption_pc[i, j] = dconsumption[i,
                                                             j] / population * 1000
        return dconsumption_pc

    def compute_dinterest_rate(self, dconsumption):
        """gradient computation for interest_rate
        Args:
            input: gradients of consumption
            output: gradients of interest_rate
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)

        # derivative matrix initialization
        dinterest_rate = np.zeros((nb_years, nb_years))

        # first line stays at zero since derivatives of initial values are zero
        for i in range(0, nb_years - 1):
            consumption = self.economics_df.at[years[i], 'consumption']
            consumption_a = self.economics_df.at[years[i + 1], 'consumption']
            for j in range(0, nb_years):
                dinterest_rate[i, j] = (1 + self.init_rate_time_pref) * (self.conso_elasticity / self.time_step) * \
                    (consumption_a / consumption)**(self.conso_elasticity / self.time_step - 1) *\
                    (dconsumption[i + 1, j] * consumption -
                     dconsumption[i, j] * consumption_a) / (consumption**2)

        return dinterest_rate

    def compute_doutput_net_of_d_damage(self, dgross_output):
        """gradient computation for output net of damages
        Args:
            input: gradients of gross output
            output: gradients of output net of damages
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)

        # derivative matrix initialization
        d_output_net_of_d = np.zeros((nb_years, nb_years))

        damage_to_productivity = self.damage_to_productivity
        frac_damage_prod = self.frac_damage_prod

        for i in range(0, nb_years):

            for j in range(0, i + 1):
                damefrac = self.damefrac.at[years[i], 'damage_frac_output']
                gross_output_ter = self.economics_df.at[years[i],
                                                        'gross_output']

                if damage_to_productivity == True:
                    damage = 1 - ((1 - damefrac) /
                                  (1 - self.frac_damage_prod * damefrac))
                    # u = 1 - damefrac
                    # v = 1 - self.frac_damage_prod * damefrac
                    d_damage = - ((- 1 * (1 - self.frac_damage_prod * damefrac) - (1 - damefrac)
                                   * (- frac_damage_prod)) / ((1 - self.frac_damage_prod * damefrac) ** 2))

                    if i == j:
                        d_output_net_of_d[i, j] = (
                            1 - damage) * dgross_output[i, j] - d_damage * gross_output_ter
                    else:
                        d_output_net_of_d[i, j] = dgross_output[i,
                                                                j] - damage * dgross_output[i, j]
                else:
                    if i == j:
                        d_output_net_of_d[i, j] = dgross_output[i, j] - \
                            gross_output_ter - dgross_output[i, j] * damefrac
                    else:
                        d_output_net_of_d[i, j] = dgross_output[i,
                                                                j] - dgross_output[i, j] * damefrac

        return d_output_net_of_d

    def compute_doutput_net_of_d(self, dgross_output):
        """gradient computation for output net of damages
        Args:
            input: gradients of gross output
            output: gradients of output net of damages
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)

        # derivative matrix initialization
        d_output_net_of_d = np.zeros((nb_years, nb_years))

        damage_to_productivity = self.damage_to_productivity
        frac_damage_prod = self.frac_damage_prod

        for i in range(0, nb_years):

            for j in range(0, i + 1):
                damefrac = self.damefrac.at[years[i], 'damage_frac_output']
                gross_output_ter = self.economics_df.at[years[i],
                                                        'gross_output']

                if damage_to_productivity == True:
                    damage = 1 - ((1 - damefrac) /
                                  (1 - self.frac_damage_prod * damefrac))
                    d_output_net_of_d[i,
                                      j] = dgross_output[i, j] * (1 - damage)
                else:
                    d_output_net_of_d[i, j] = dgross_output[i,
                                                            j] * (1 - damefrac)

        return d_output_net_of_d

    """-------------------END of Gradient functions-------------------"""

    def compute(self, inputs, damage_prod=False):
        """
        Compute all models for year range
        """
        self.create_dataframe()
        self.damage_prod = damage_prod
        self.inputs = inputs
#         self.abatecost = self.inputs['abatecost']
        self.damefrac = self.inputs['damage_frac_output']
        self.damefrac.index = self.damefrac['years'].values
        self.scaling_factor_energy_production = self.inputs['scaling_factor_energy_production']
        self.scaling_factor_energy_investment = self.inputs['scaling_factor_energy_investment']
        self.energy_production = self.inputs['energy_production'].copy(
            deep=True)
        self.energy_production['Total production'] *= self.scaling_factor_energy_production
        self.co2_emissions_Gt = pd.DataFrame({'years': self.inputs['co2_emissions_Gt']['years'].values,
                                              'Total CO2 emissions': self.inputs['co2_emissions_Gt']['Total CO2 emissions'].values})
        self.co2_emissions_Gt.index = self.co2_emissions_Gt['years'].values
        self.co2_taxes = self.inputs['CO2_taxes']
        self.co2_taxes.index = self.co2_taxes['years'].values

        self.energy_production.index = self.energy_production['years'].values
        self.share_energy_investment = pd.Series(
            self.inputs['share_energy_investment']['share_investment'].values / 100.0, index=self.years_range)
        self.total_share_investment = pd.Series(
            self.inputs['total_investment_share_of_gdp']['share_investment'].values / 100.0, index=self.years_range)
#       self.share_n_energy_investment = pd.Series(list(self.inputs['share_non_energy_investment']['share_investment']), index=(np.arange(
#         self.param['year_start'], self.param['year_end'] + 1,
#         self.param['time_step'])))
        self.share_n_energy_investment = self.total_share_investment - \
            self.share_energy_investment
        self.population_df = self.inputs['population_df']
        self.population_df.index = self.population_df['years'].values
        # Check that invest below limit
#         invest_share = list(self.share_n_energy_investment +
#                             self.share_energy_investment)
#         count = len([i for i in invest_share if i > self.max_invest])
#         if count > 0:
#             print(
#                 'For at least one year, the total investment (energy + non energy) is above max_invest')
        # YEAR START
        self.compute_output_net_of_damage(self.year_start)
        self.compute_net_output(self.year_start)
        self.compute_energy_investment(self.year_start)
        self.compute_investment(self.year_start)
        self.compute_consumption(self.year_start)
        self.compute_consumption_pc(self.year_start)
        # for year 0 compute capital +1
        self.compute_capital(self.year_start)
        # Then iterate over years from year_start + tstep:
        for year in self.years_range[1:]:
            # First independant variables
            self.compute_productivity_growthrate(year)
            self.compute_productivity(year)
            self.compute_energy_productivity_growthrate(year)
            self.compute_energy_productivity(year)
            # Then others:
            self.compute_gross_output(year)
            self.compute_output_net_of_damage(year)
            self.compute_net_output(year)
            self.compute_energy_investment(year)
            self.compute_investment(year)
            self.compute_consumption(year)
            self.compute_consumption_pc(year)
            # capital t+1 :
            self.compute_capital(year)
        # Then interest rate
        for year in self.years_range:
            # self.compute_interest_rate(year)
            self.compute_output_growth(year)
        self.economics_df = self.economics_df.replace(
            [np.inf, -np.inf], np.nan)
        # Compute consumption per capita constraint
        self.compute_comsumption_pc_constraint()
        # Compute global investment constraint
        self.global_investment_constraint = deepcopy(
            self.inputs['share_energy_investment'])

        self.global_investment_constraint['share_investment'] = self.global_investment_constraint['share_investment'].values / 100.0 + \
            self.share_n_energy_investment.values - \
            self.inputs['total_investment_share_of_gdp']['share_investment'].values / 100.0

        return self.economics_df.fillna(0.0), self.energy_investment.fillna(0.0), self.global_investment_constraint, \
            self.energy_investment_wo_renewable.fillna(
                0.0), self.pc_consumption_constraint
