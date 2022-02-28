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
from itertools import chain
from pandas import DataFrame, concat
import numpy as np
from copy import deepcopy


class Population:
    """
    Population model mostly based on McIsaac, F., 2020. A Representation of the World Population Dynamics for Integrated Assessment Models.
     Environmental Modeling & Assessment, 25(5), pp.611-632.
    """

    def __init__(self, inputs):
        '''
        Constructor
        '''
        self.population_df = None
        self.working_population_df = None
        self.birth_rate = None
        self.death_rate_df = None
        self.set_data(inputs)
        self.trillion = 1e12
        self.billion = 1e9
        self.million = 1e6

    def set_data(self, inputs):

        self.year_start = inputs['year_start']
        self.year_end = inputs['year_end']
        self.time_step = inputs['time_step']
        self.pop_init_df = inputs['population_start']
        self.br_upper = inputs['birth_rate_upper']
        self.br_lower = inputs['birth_rate_lower']
        self.br_phi = inputs['birth_rate_phi']
        self.br_nu = inputs['birth_rate_nu']
        self.br_delta = inputs['birth_rate_delta']
        self.climate_mortality_param_df = inputs['climate_mortality_param_df']
        self.cal_temp_increase = inputs['calibration_temperature_increase']
        self.theta = inputs['theta']
        self.dr_param_df = inputs['death_rate_param']
        # Age range list for death rate is the same as the one from param df
        self.age_list = list(self.dr_param_df['param'])
        self.lower_know = inputs['lower_knowledge']
        self.upper_know = inputs['upper_knowledge']
        self.delta_know = inputs['delta_knowledge']
        self.phi_know = inputs['phi_knowledge']
        self.nu_know = inputs['nu_knowledge']
        self.cst_br_k = inputs['constant_birthrate_know']
        self.alpha_br_k = inputs['alpha_birthrate_know']
        self.beta_br_k = inputs['beta_birthrate_know']
        self.share_know = inputs['share_know_birthrate']
        # First year of the regression of knowledge function
        self.year_reg_know = 1800


    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        years_range = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years_range = years_range
        # Prepare columns of population df
        pop_column = [str(x) for x in np.arange(0, 100)]
        pop_column.append('100+')
        self.full_age_list = pop_column
        pop_df_column = pop_column.copy()
        pop_df_column.insert(0, 'years')
        pop_df_column.append('total')
        population_df = DataFrame(
            index=years_range,
            columns=pop_df_column)
        for key in population_df.keys():
            population_df[key] = 0
        population_df.loc[self.year_start,
                          'total'] = self.pop_init_df['population'].sum()
        # POPULATION
        # select all age except 100+
        pop_init = self.pop_init_df.iloc[:-1, 1]
        # Divide by 5
        pop_init = list(pop_init.divide(other=5))
        # Duplicate over age group
        pop_by_age = list(chain(*zip(*[pop_init for _ in range(5)])))
        # Add 100+ value of pop init
        pop_by_age.append(self.pop_init_df.iloc[-1, 1])
        population_df.iloc[0, 1:-1] = pop_by_age
        population_df['years'] = years_range
        self.population_df = population_df

        column_list = self.age_list.copy()
        column_list.insert(0, 'years')

        # WORKING POULATION
        self.working_population_df = DataFrame(index=years_range,
                                    columns=['years', 'population_1570'])
        self.working_population_df['years'] = years_range
        self.working_population_df.loc[self.year_start,
                          'population_1570'] = population_df.loc[self.year_start, '15':'70'].sum()#will take 15yo to 70yo

        # BIRTH RATE
        # BASE => calculated from GDB and knowledge level
        self.birth_rate = DataFrame(index=years_range,
                                    columns=['years', 'knowledge', 'birth_rate'])
        for key in self.birth_rate.keys():
            self.birth_rate[key] = 0
        self.birth_rate['years'] = years_range

        # BIRTH NUMBER
        # BASE => calculated from GDB and knowledge level
        self.birth_df = DataFrame(index=years_range,
                                  columns=['years', 'knowledge', 'birth_rate'])
        for key in self.birth_df.keys():
            self.birth_df[key] = 0
        self.birth_df['years'] = years_range

        # DEATH RATE - slices of 4 years age in column
        # BASE => calculated from GDP
        self.base_death_rate_df = DataFrame(index=years_range,
                                            columns=column_list)
        for key in self.base_death_rate_df.keys():
            self.base_death_rate_df[key] = 0
        self.base_death_rate_df['years'] = years_range

        # CLIMATE => calculated from temperature increase
        self.climate_death_rate_df = DataFrame(index=years_range,
                                               columns=column_list)
        for key in self.climate_death_rate_df.keys():
            self.climate_death_rate_df[key] = 0
        self.climate_death_rate_df['years'] = years_range

        # TOTAL => sum of all effects
        self.death_rate_df = DataFrame(index=years_range,
                                       columns=column_list)
        for key in self.death_rate_df.keys():
            self.death_rate_df[key] = 0
        self.death_rate_df['years'] = years_range

        # CONTAINER => dictionnary containing death rates
        self.death_rate_dict = {'base': self.base_death_rate_df, 'climate': self.climate_death_rate_df,
                                'total': self.death_rate_df}

        # DEATH NUMBER - one column per age
        # BASE => calculated from GDP
        self.base_death_df = DataFrame(index=years_range,
                                       columns=pop_df_column)
        for key in self.base_death_df.keys():
            self.base_death_df[key] = 0
        self.base_death_df['years'] = years_range

        # CLIMATE => calculated from temperature increase
        self.climate_death_df = DataFrame(index=years_range,
                                          columns=pop_df_column)
        for key in self.climate_death_df.keys():
            self.climate_death_df[key] = 0
        self.climate_death_df['years'] = years_range

        # TOTAL => sum of all effects
        self.death_df = DataFrame(index=years_range,
                                  columns=pop_df_column)
        for key in self.death_df.keys():
            self.death_df[key] = 0
        self.death_df['years'] = years_range

        # CONTAINER => dictionnary containing death number
        self.death_dict = {'base': self.base_death_df,
                           'climate': self.climate_death_df, 'total': self.death_df}

        # LIFE EXPECTANCY
        self.life_expectancy_df = DataFrame(index=years_range,
                                            columns=['years', 'life_expectancy'])
        for key in self.life_expectancy_df.keys():
            self.life_expectancy_df[key] = 0
        self.life_expectancy_df['years'] = years_range

        return population_df

    def compute_knowledge(self):
        """ Compute knowledge function for all year. Knowledge is a regression on % of 
            litterate world pop. 
        """
        x = self.years_range - self.year_reg_know
        knowledge = self.lower_know + (self.upper_know - self.lower_know) \
            * (1 / (1 + np.exp(-self.delta_know * (x - self.phi_know))) ** self.nu_know)
        self.birth_rate['knowledge'] = knowledge

        return knowledge

    def compute_birth_rate_v1(self, year):
        ''' Compute the birth rate. The birth rate can be defined as birth_rate = number of born/pop_1549
        Inputs : - economics df: dataframe containing the economic output/ gdp per year in trillions $ 
                 - population_df: dataframe containing total number of population per year in nb of people
                 - year: the year for which we want to estimate the value of the birth rate
                 - parameters of the function: br_lower_a, br_upper_a, br_delta, br_nu, br_phi
        output : birth rate for the year 
        '''
        # COnvert GDP in $
        gdp = self.economics_df.loc[year, 'output_net_of_d'] * self.trillion
        pop = self.population_df.loc[year, 'total']
        birth_rate = self.br_upper + (self.br_lower - self.br_upper) / (
            1 + np.exp(-self.br_delta * (gdp / pop - self.br_phi))) ** (1 / self.br_nu)
        self.birth_rate.loc[year, 'birth_rate'] = birth_rate

        return birth_rate

    def compute_birth_rate_v2(self, year):
        """ Compute birth rate. birth rate = a * f(knowledge) + (1-a)*f(gdp)
        all parameters obtained by fitting of birth rate data btwn 1960 and 2020
        Inputs: knowledge (series per year), gdp (series per year, pop (series per year), params
        """
        # Convert GDP in $
        gdp = self.economics_df.loc[year, 'output_net_of_d'] * self.trillion
        pop = self.population_df.loc[year, 'total']
        knowledge = self.birth_rate.loc[year, 'knowledge']
        # Compute in two steps
        f_knowledge = self.cst_br_k + self.alpha_br_k * \
            (1 - knowledge / 100) ** self.beta_br_k
        f_gdp = self.br_upper + (self.br_lower - self.br_upper) / (
            1 + np.exp(-self.br_delta * (gdp / pop - self.br_phi))) ** (1 / self.br_nu)
        birth_rate = self.share_know * f_knowledge + \
            (1 - self.share_know) * f_gdp

        self.birth_rate.loc[year, 'birth_rate'] = birth_rate

        return birth_rate

    def compute_death_rate(self, year):
        ''' Compute the death rate for each age range. The birth rate can be defined as 
            death_rate = number of death/pop_agerange
        Inputs : - economics df: dataframe containing the economic output/ gdp per year in trillions $ 
                 - population_df: dataframe containing total number of population per year in nb of people
                 - year: the year for which we want to estimate the value of the birth rate
                 - parameters of the function: 
        output : death rate for the year for each age range. type: pandas Series   
        '''
        gdp = self.economics_df.loc[year, 'output_net_of_d'] * self.trillion
        pop = self.population_df.loc[year, 'total']
        param = self.dr_param_df
        # For all age range compute death rate
        death_rate = param['death_rate_upper'] + (param['death_rate_lower'] - param['death_rate_upper']) / (
            1 + np.exp(-param['death_rate_delta'] *
                       (gdp / pop - param['death_rate_phi']))) ** (1 / param['death_rate_nu'])
        # Fill the year row in death rate df
        self.death_rate_df.iloc[year - self.year_start, 1:] = death_rate

        return self.death_rate_df.loc[year]

    def compute_death_rate_v2(self, year):
        ''' Compute the death rate for each age range. The birth rate can be defined as 
            death_rate = number of death/pop_agerange
        Inputs : - economics df: dataframe containing the economic output/ gdp per year in trillions $
                 - temperature df: dataframe containing the temperature increase per year since preindustrial era
                 - population_df: dataframe containing total number of population per year in nb of people
                 - year: the year for which we want to estimate the value of the birth rate
                 - parameters of the function: 
        output : death rate for the year for each age range. type: pandas Series   
        '''
        gdp = self.economics_df.at[year, 'output_net_of_d'] * self.trillion
        pop = self.population_df.loc[year, 'total']
        temp = self.temperature_df.loc[year, 'temp_atmo']
        param = self.dr_param_df
        add_death = self.climate_mortality_param_df
        cal_temp_increase = self.cal_temp_increase
        theta = self.theta
        # For all age range compute death rate
        death_rate = param['death_rate_upper'] + (param['death_rate_lower'] - param['death_rate_upper']) / (
            1 + np.exp(-param['death_rate_delta'] *
                       (gdp / pop - param['death_rate_phi']))) ** (1 / param['death_rate_nu'])
        # Add climate impact on death rate
        climate_death_rate = add_death['beta'] * \
            (temp / cal_temp_increase) ** theta

        # Fill the year row in death rate df
        self.base_death_rate_df.iloc[year - self.year_start, 1:] = death_rate
        self.climate_death_rate_df.iloc[year - self.year_start,
                                        1:] = climate_death_rate * death_rate
        self.death_rate_df.iloc[year - self.year_start,
                                1:] = death_rate * (1 + climate_death_rate)

        return self.death_rate_dict

    def compute_death_number(self, year):
        """Compute number of dead people per year
        input: population df
                death rate value per age range 
        output df of number of death 
        """
        # Pop year = row of df without year and total
        pop_year = self.population_df.iloc[year - self.year_start, 1:-1]
        total_deaths = deepcopy(pop_year)
        for key in total_deaths.keys():
            total_deaths[key] = 0.0

        for effect in self.death_rate_dict:
            if effect != 'total':
                dr_year = self.death_rate_dict[effect].iloc[year -
                                                            self.year_start, 1:-1]
                # Duplicate each element of the list 5 times so that we have a
                # death rate per age
                full_dr_death = list(
                    chain(*zip(*[list(dr_year) for _ in range(5)])))
                full_dr_death.append(
                    self.death_rate_dict[effect].loc[year, '100+'])
                nb_death = pop_year.multiply(full_dr_death)

                total_deaths += nb_death
                self.death_dict[effect].iloc[year -
                                             self.year_start, 1:-1] = list(nb_death)
            else:
                self.death_dict[effect].iloc[year -
                                             self.year_start, 1:-1] = total_deaths

        return total_deaths

    def compute_birth_number(self, year):
        '''Compute number of birth per year
        input: birth rate 
                population df 
        output df of number of birth per year
        '''
        # Sum population between 15 and 49year
        pop_1549 = sum(self.population_df.loc[year, '15':'49'])
        nb_birth = self.birth_rate.at[year, 'birth_rate'] * pop_1549
        self.birth_df.loc[year, 'number_of_birth'] = nb_birth

        return nb_birth

    def compute_population_next_year(self, year, total_death, nb_birth):
        if year > self.year_end:
            pass
        else:
            year_start = self.year_start
            pop_before = self.population_df.iloc[year - 1 - year_start, 1:-1]
            pop_before -= total_death

            # Add new born
            self.population_df.loc[year, '0'] = nb_birth
            # Then update population
            # And +1 for each person alive
            self.population_df.iloc[year - year_start, 2:-1] = pop_before[:-1]
            # Add not dead people over 100+
            old_not_dead = pop_before[-1]
            self.population_df.loc[year, '100+'] += old_not_dead
            self.population_df.loc[year,
                                   'total'] = self.population_df.iloc[year - year_start, 1:-1].sum()
            # compute working population from 15yo to 70yo
            self.working_population_df.loc[year, 'population_1570'] = sum(self.population_df.loc[year, '15':'70'])

        return self.population_df


    def compute_life_expectancy(self, year):
        """
        Compute life expectancy for a year
        life expectancy = sum(pop_i) with i the age 
        pop_0 = 1 
        pop_i = pop_i-1(1- death_rate_i) 
        """
        dr_year = self.death_rate_df.iloc[year - self.year_start, 1:-1]
        # Duplicate each element of the list 5 times so that we have a death
        # rate per age
        full_dr_death = list(chain(*zip(*[list(dr_year) for _ in range(5)])))
        full_dr_death.append(self.death_rate_df.loc[year, '100+'])

        nb_age = len(self.full_age_list)
        pop = [0] * nb_age
        # Start with a pop = 1 and iterate over ages
        pop[0] = 1
        # COmpute surviving people
        for i in range(0, nb_age - 1):
            pop[i + 1] = pop[i] * (1 - full_dr_death[i])
        # Sum all surviving people and divide by the initial pop = 1
        life_expectancy = np.sum(pop)
        self.life_expectancy_df.loc[year, 'life_expectancy'] = life_expectancy

        return self.life_expectancy_df

    def compute(self, in_dict):
        """
        Compute all
        """
        self.create_dataframe()
        year_range = self.years_range
        self.economics_df = in_dict['economics_df']
        self.economics_df.index = self.economics_df['years'].values
        self.temperature_df = in_dict['temperature_df']
        self.temperature_df.index = self.temperature_df['years'].values
        self.compute_knowledge()
        # Loop over year to compute population evolution. except last year
        for year in year_range:
            self.compute_birth_rate_v2(year)
            self.compute_death_rate_v2(year)
            total_death = self.compute_death_number(year)
            nb_birth = self.compute_birth_number(year)
            self.compute_population_next_year(year + 1, total_death, nb_birth)
            self.compute_life_expectancy(year)
        self.population_df = self.population_df.replace(
            [np.inf, -np.inf], np.nan)

        # Calculation of cumulative deaths
        for effect in self.death_dict:
            cumulative_death = DataFrame()
            self.death_dict[effect]['total'] = self.death_dict[effect].iloc[:,
                                                                            1:-1].sum(axis=1, skipna=True)
            cumulative_death['cum_total'] = self.death_dict[effect]['total'].cumsum()
            self.death_dict[effect] = concat([self.death_dict[effect],cumulative_death], axis=1)
        
        for effect in self.death_dict:
            self.death_dict[effect].fillna(0.0)
            self.death_rate_dict[effect].fillna(0.0)

        return self.population_df.fillna(0.0), self.birth_rate.fillna(0.0), self.death_rate_dict, \
            self.birth_df.fillna(
                0.0), self.death_dict, self.life_expectancy_df.fillna(0.0), self.working_population_df.fillna(0.0)

    # GRADIENTS OF POPULATION WTR GDP
    def compute_d_pop_d_output(self):
        """ Compute the derivative of population wrt output
        """
        nb_years = (self.year_end - self.year_start + 1)

        # derivative of population of the current year for each age
        d_pop_d_output = {}
        d_working_pop_d_output = np.zeros((nb_years, nb_years))
        d_pop_tot_d_output = np.zeros((nb_years, nb_years))
        d_birthrate_d_output = np.zeros((nb_years, nb_years))
        d_birth_d_output = np.zeros((nb_years, nb_years))
        d_base_deathrate_d_output = {}
        d_death_rate_climate_d_output = {}
        d_death_d_output = {}
        d_pop_1549_d_output = np.zeros((nb_years, nb_years))

        for year in self.years_range:
            if year == self.year_end:
                pass
            else:
                d_birthrate_d_output += self.d_birthrate_d_output(
                    year, d_pop_tot_d_output)
                d_birth_d_output += self.d_birth_d_generic(
                    year, d_birthrate_d_output, d_pop_1549_d_output)

                d_base_deathrate_d_output[year] = self.d_base_death_rate_d_output(
                    year, d_pop_tot_d_output)
                d_death_rate_climate_d_output[year] = self.d_climate_death_rate_d_output(
                    year, d_base_deathrate_d_output)

                d_death_d_output[year] = self.d_death_d_generic(year, d_base_deathrate_d_output,
                                                                d_death_rate_climate_d_output, d_pop_d_output)
                d_pop_d_output, d_pop_1549_d_output, d_pop_tot_d_output, d_working_pop_d_output = self.d_poptotal_generic(year, d_pop_d_output,
                                                                                                  d_death_d_output,
                                                                                                  d_birth_d_output,
                                                                                                  d_pop_1549_d_output,
                                                                                                  d_pop_tot_d_output,
                                                                                                  d_working_pop_d_output)
        return d_pop_tot_d_output, d_working_pop_d_output

    def d_birthrate_d_output(self, year, d_pop_tot_d_output):
        """ Compute the derivative of birth rate wrt output
        f_knowledge = self.cst_br_k + self.alpha_br_k * (1- knowledge/100)**self.beta_br_k
        f_gdp = self.br_upper + (self.br_lower - self.br_upper)/(1+np.exp(-self.br_delta*(gdp/pop -self.br_phi)))**(1/self.br_nu)
        birth_rate = self.share_know * f_knowledge + (1-self.share_know) * f_gdp
        """

        nb_years = (self.year_end - self.year_start + 1)
        d_birthrate_d_output = np.zeros((nb_years, nb_years))
        iyear = year - self.year_start
        idty = np.zeros(nb_years)
        idty[iyear] = 1

        # Param of birth rate equation
        delta = self.br_delta
        phi = self.br_phi
        br_upper = self.br_upper
        br_lower = self.br_lower
        nu = self.br_nu
        pop = self.population_df.loc[year, 'total']
        gdp = self.economics_df.loc[year, 'output_net_of_d'] * self.trillion
        d_pop = d_pop_tot_d_output[iyear]

        # f gdp = br_upper + (self.br_lower - self.br_upper)/ u
        # -> fprime = (self.br_lower - self.br_upper)*u_prime/u_squared
        u_squared = ((1 + np.exp(-delta * (gdp / pop - phi))) ** (1 / nu)) ** 2
        # u = g(f(x))  #u = g(f(x)) with g = f**(1:nu) and f = 1+
        # exp(-delta(gdp/pop-phi))
        g_prime_f = (1 / nu) * (1 + np.exp(-delta *
                                           (gdp / pop - phi))) ** (1 / nu - 1)
        f_prime = -delta * (pop * idty * self.trillion - d_pop * gdp) / (pop ** 2) * (
            np.exp(-delta * (gdp / pop - phi)))

        u_prime = g_prime_f * f_prime
        d_f_gdp_d_output = - (br_lower - br_upper) * u_prime / u_squared

        # and lastly multiply by (1-share now)
        d_birthrate_d_output[iyear] = (1 - self.share_know) * d_f_gdp_d_output

        return d_birthrate_d_output

    def d_birth_d_generic(self, year, d_birthrate_d_output, d_pop_1549_d_output):
        """ Compute the derivative of birth number wrt output
        nb birth = pop * birth_rate
        """
        nb_years = (self.year_end - self.year_start + 1)
        iyear = year - self.year_start
        idty = np.zeros(nb_years)
        idty[iyear] = 1
        d_birth_d_output = np.zeros((nb_years, nb_years))
        pop_1549 = sum(self.population_df.iloc[iyear, 16:51])
        d_pop_1549 = d_pop_1549_d_output[iyear]

        br = self.birth_rate.at[year, 'birth_rate']

        # nb_birth = pop_1549 * birth_rate => d_nb_birth = u'v + v'u
        d_birth_d_output[iyear] = d_birthrate_d_output[iyear] * \
            pop_1549 + d_pop_1549 * br

        return d_birth_d_output

    def d_base_death_rate_d_output(self, year, d_pop_tot_d_output):
        """
        for every columns of death rate dataframe create a dictionary that contains all derivatives wrt output
        """
        nb_years = (self.year_end - self.year_start + 1)
        iyear = year - self.year_start
        idty = np.zeros(nb_years)
        idty[iyear] = 1

        param = self.dr_param_df
        param.index = param['param'].values

        d_deathrate_d_output = {}
        pop = self.population_df.loc[year, 'total']
        gdp = self.economics_df.loc[year, 'output_net_of_d'] * self.trillion
        d_pop = d_pop_tot_d_output[iyear]

        for age_range in param['param'].values:
            # Param of death rate equation
            delta = param.loc[age_range, 'death_rate_delta']
            phi = param.loc[age_range, 'death_rate_phi']
            br_upper = param.loc[age_range, 'death_rate_upper']
            br_lower = param.loc[age_range, 'death_rate_lower']
            nu = param.loc[age_range, 'death_rate_nu']
            # Value on the diagonal. death rate t depends on output t
            # derivative = (lower-upper)*u'/u^2
            u_squared = ((1 + np.exp(-delta * (gdp / pop - phi)))
                         ** (1 / nu)) ** 2
            # u = g(f(x)) with g = f**(1:nu) and f = (1+
            # exp(-delta(gdp/pop-phi))
            g_prime_f = (1 / nu) * (1 + np.exp(-delta *
                                               (gdp / pop - phi))) ** (1 / nu - 1)
            f_prime = -delta * (pop * idty * self.trillion - d_pop * gdp) / (pop ** 2) * np.exp(
                -delta * (gdp / pop - phi))
            u_prime = g_prime_f * f_prime

            d_death_rate_d_output_age = - \
                (br_lower - br_upper) * u_prime / u_squared
            d_deathrate_d_output[age_range] = d_death_rate_d_output_age
        return d_deathrate_d_output

    def d_climate_death_rate_d_output(self, year, d_base_death_rate):
        """
        for every columns of death rate dataframe create a dictionary that contains all derivatives wrt temp
        """
        nb_years = (self.year_end - self.year_start + 1)
        iyear = year - self.year_start
        idty = np.zeros(nb_years)
        idty[iyear] = 1
        param = self.dr_param_df
        param.index = param['param'].values

        d_climate_deathrate_d_output = {}
        climate_death_rate = self.climate_death_rate_df.iloc[iyear, 1:]
        temp = self.temperature_df.loc[year, 'temp_atmo']
        add_death = self.climate_mortality_param_df
        add_death.index = param['param'].values
        cal_temp_increase = self.cal_temp_increase
        theta = self.theta

        for age_range in param['param'].values:
            # (uv)' = u_prime_v * v_prime_u with u = base_deathrate v = climate_deathrate
            # v_prime_u = 0 because d_climate_death_rate = 0
            climate_death_rate = add_death['beta'][age_range] * \
                (temp / cal_temp_increase) ** theta
            u_prime_v = d_base_death_rate[year][age_range] * climate_death_rate
            d_climate_deathrate_d_output[age_range] = u_prime_v

        return d_climate_deathrate_d_output

    def d_death_d_generic(self, year, dict_d_base_death_rate_d_temp, dict_d_climate_death_rate_d_temp,
                          dict_d_population_d_output):
        """
        Compute derivative of each column of death df wrt output and returns a dictionary

        """
        iyear = year - self.year_start
        number_of_values = (self.year_end - self.year_start + 1)
        d_death = {}
        idty = np.zeros(number_of_values)
        idty[iyear] = 1

        pop_year = self.population_df.iloc[iyear, 1:-1]
        ages = self.population_df.columns[1:-1]

        # step one get values from dictionaries
        list_d_base_dr_d_out = list(
            dict_d_base_death_rate_d_temp[year].values())
        list_d_climate_dr_d_out = list(
            dict_d_climate_death_rate_d_temp[year].values())

        list_d_pop_d_out = {}
        if year in dict_d_population_d_output.keys():
            list_d_pop_d_out = dict_d_population_d_output[year]

        # get death rate by age
        dr_year = self.death_rate_dict['total'].iloc[iyear, 1:-1]
        # Duplicate each element of the list 5 times so that we have a death
        # rate per age
        full_dr_death = list(chain(*zip(*[list(dr_year) for _ in range(5)])))
        full_dr_death.append(self.death_rate_dict['total'].loc[year, '100+'])

        # Remove 100+ value
        base_value_hundred = list_d_base_dr_d_out[-1]
        base_list_d_dr_d_out_t = list_d_base_dr_d_out[:-1]
        # Duplicate to get one per age
        base_list_d_dr_d_out = list(
            chain(*zip(*[base_list_d_dr_d_out_t for _ in range(5)])))
        # and add value of last year
        base_list_d_dr_d_out.append(base_value_hundred)

        # Remove 100+ value
        climate_value_hundred = list_d_climate_dr_d_out[-1]
        climate_list_d_dr_d_out_t = list_d_climate_dr_d_out[:-1]
        # Duplicate to get one per age
        climate_list_d_dr_d_out = list(
            chain(*zip(*[climate_list_d_dr_d_out_t for _ in range(5)])))
        # and add value of last year
        climate_list_d_dr_d_out.append(climate_value_hundred)
        # Compute derivative for each column of dataframe

        for i in range(0, len(ages)):
            if ages[i] not in list_d_pop_d_out.keys():
                list_d_pop_d_out[ages[i]] = np.zeros(number_of_values)
            d_death[ages[i]] = list_d_pop_d_out[ages[i]] * full_dr_death[i] + \
                (climate_list_d_dr_d_out[i] +
                 base_list_d_dr_d_out[i]) * pop_year[i]

        return d_death

    def d_poptotal_generic(self, year, d_pop, d_death, d_birth, d_pop_1549,
                           d_total_pop, d_working_pop):
        """
        Compute derivative of column total of pop df wrt output
        """
        # Derivative of a sum = sum of derivatives
        if year + 1 > self.year_end:
            pass
        else:
            iyear = year - self.year_start
            number_of_values = (self.year_end - self.year_start + 1)
            idty = np.zeros(number_of_values)
            idty[iyear] = 1

            d_pop_d_y = {}
            sum_tot_pop = np.zeros(number_of_values)
            age_list = self.population_df.columns[1:-1]
            range_age_1549 = np.arange(15, 49+1)# 15 to 49
            range_age_1570 = np.arange(15, 70+1)# 15 to 70
            d_pop_d_age_prev = {}
            sum_pop_1549 = np.zeros(number_of_values)
            sum_pop_1570 = np.zeros(number_of_values)

            for i in range(0, len(age_list)):
                # compute population of previous year: population - nb_death =>
                # derivative = d_pop - d_nb_death
                if year in d_pop.keys():
                    d_pop_d_age_prev[age_list[i]] = d_pop[year][age_list[i]] - d_death[year][
                        age_list[i]]
                else:
                    d_pop_d_age_prev[age_list[i]] = -d_death[year][age_list[i]]

                d_pop_d_y[age_list[i]] = np.zeros(number_of_values)

                if i == 0:
                    # at age = 0 pop = nb_birth => d_pop = d_nb_birth
                    d_pop_d_y[age_list[i]] = d_birth[iyear]
                else:
                    # at year age between 1 and 100+: = pop_before =>
                    # derivative = d_pop_before
                    d_pop_d_y[age_list[i]] = d_pop_d_age_prev[age_list[i - 1]]
                sum_tot_pop += d_pop_d_y[age_list[i]]
                if i in range_age_1549:
                    sum_pop_1549 += d_pop_d_y[age_list[i]]
                if i in range_age_1570:
                    sum_pop_1570 += d_pop_d_y[age_list[i]]

            # add old not dead at year before at 100+ this year
            d_old_not_dead = d_pop_d_age_prev[age_list[-1]]
            d_pop_d_y[age_list[-1]] += d_old_not_dead
            sum_tot_pop += d_old_not_dead

            d_pop[year + 1] = d_pop_d_y
            d_pop_1549[iyear + 1] = sum_pop_1549
            d_working_pop[iyear + 1] = sum_pop_1570
            d_total_pop[iyear + 1] = sum_tot_pop

        return d_pop, d_pop_1549, d_total_pop, d_working_pop

    # WRT TEMPERATURE
    def compute_d_pop_d_temp(self):
        """ Compute the derivative of population wrt temp
        """
        nb_years = (self.year_end - self.year_start + 1)

        # derivative of population of the current year for each age
        d_pop_d_temp = {}
        d_working_pop_d_temp = np.zeros((nb_years, nb_years))
        d_pop_tot_d_temp = np.zeros((nb_years, nb_years))
        d_birthrate_d_temp = np.zeros((nb_years, nb_years))
        d_birth_d_temp = np.zeros((nb_years, nb_years))
        d_base_death_rate = {}
        d_climate_death_rate = {}
        d_death_d_temp = {}
        d_pop_1549_d_temp = np.zeros((nb_years, nb_years))

        for year in self.years_range:
            if year == self.year_end:
                pass
            else:
                d_birthrate_d_temp += self.d_birthrate_d_temp(
                    year, d_pop_tot_d_temp)
                d_birth_d_temp += self.d_birth_d_generic(
                    year, d_birthrate_d_temp, d_pop_1549_d_temp)

                d_base_death_rate[year] = self.d_base_death_rate_d_temp(
                    year, d_pop_tot_d_temp)
                d_climate_death_rate[year] = self.d_climate_death_rate_d_temp(
                    year, d_base_death_rate)

                d_death_d_temp[year] = self.d_death_d_generic(
                    year, d_base_death_rate, d_climate_death_rate, d_pop_d_temp)
                d_pop_d_temp, d_pop_1549_d_temp, d_pop_tot_d_temp, d_working_pop_d_temp = self.d_poptotal_generic(year, d_pop_d_temp,
                                                                                            d_death_d_temp,
                                                                                            d_birth_d_temp,
                                                                                            d_pop_1549_d_temp,
                                                                                            d_pop_tot_d_temp,
                                                                                            d_working_pop_d_temp)
        return d_pop_tot_d_temp, d_working_pop_d_temp

    def d_birthrate_d_temp(self, year, d_pop_tot_d_temp):
        """ Compute the derivative of birth rate wrt temp
        f_knowledge = self.cst_br_k + self.alpha_br_k * (1- knowledge/100)**self.beta_br_k
        f_gdp = self.br_upper + (self.br_lower - self.br_upper)/(1+np.exp(-self.br_delta*(gdp/pop -self.br_phi)))**(1/self.br_nu)
        birth_rate = self.share_know * f_knowledge + (1-self.share_know) * f_gdp
        """
        nb_years = (self.year_end - self.year_start + 1)
        d_birthrate_d_temp = np.zeros((nb_years, nb_years))
        iyear = year - self.year_start
        # Param of birth rate equation
        delta = self.br_delta
        phi = self.br_phi
        br_upper = self.br_upper
        br_lower = self.br_lower
        nu = self.br_nu
        pop = self.population_df.loc[year, 'total']
        gdp = self.economics_df.loc[year, 'output_net_of_d'] * self.trillion
        d_pop = d_pop_tot_d_temp[iyear]
        idty = np.zeros(nb_years)
        idty[iyear] = 1
        # f gdp = br_upper + (self.br_lower - self.br_upper)/ u
        # -> fprime = (self.br_lower - self.br_upper)*u_prime/u_squared
        u_squared = ((1 + np.exp(- delta * (gdp / pop - phi)))
                     ** (1 / nu)) ** 2
        # u = g(f(x))  #u = g(f(x)) with g = f**(1/nu) and f = 1+
        # exp(-delta(gdp/pop-phi))
        g_prime_f = (1 / nu) * (1 + np.exp(- delta *
                                           (gdp / pop - phi))) ** (1 / nu - 1)
        f_prime = -delta * (- gdp * d_pop) / (pop ** 2) * \
            np.exp(-delta * (gdp / pop - phi))
        u_prime = g_prime_f * f_prime
        d_f_gdp_d_temp = - (br_lower - br_upper) * u_prime / u_squared

        d_f_gdp_d_net_temp = d_f_gdp_d_temp
        # and lastly multiply by (1-share now)
        d_birthrate_d_temp[iyear] = (1 - self.share_know) * d_f_gdp_d_net_temp

        return d_birthrate_d_temp

    def d_base_death_rate_d_temp(self, year, d_pop_tot_d_temp):
        """
        for every columns of death rate dataframe create a dictionary that contains all derivatives wrt temp
        """
        nb_years = (self.year_end - self.year_start + 1)
        iyear = year - self.year_start
        idty = np.zeros(nb_years)
        idty[iyear] = 1

        param = self.dr_param_df
        param.index = param['param'].values

        d_base_deathrate_d_temp = {}
        pop = self.population_df.loc[year, 'total']
        gdp = self.economics_df.loc[year, 'output_net_of_d'] * self.trillion
        add_death = self.climate_mortality_param_df
        add_death.index = add_death['param'].values

        d_pop = d_pop_tot_d_temp[iyear]

        for age_range in param['param'].values:
            # Param of death rate equation
            delta = param.loc[age_range, 'death_rate_delta']
            phi = param.loc[age_range, 'death_rate_phi']
            dr_upper = param.loc[age_range, 'death_rate_upper']
            dr_lower = param.loc[age_range, 'death_rate_lower']
            nu = param.loc[age_range, 'death_rate_nu']
            # Value on the diagonal. death rate t depends on temp t
            # dr = u + u * v
            # u = upper + (lower - upper) / (1 + np.exp(-delta * (gdp / pop - phi))) ** (1 / nu)
            # v = beta * (temp / cal_temp_increase) ** theta
            # u' = (lower-upper)*w'/w^2
            # w = g(f(x)) with g = f**(1:nu) and f = (1+ exp(-delta(gdp/pop-phi))
            # v' = beta * tetha / cal_temp_increase * (temp /
            # cal_temp_increase) ** (theta - 1)
            w_squared = ((1 + np.exp(-delta * (gdp / pop - phi)))
                         ** (1 / nu)) ** 2
            # w = g(f(x)) with g = f**(1:nu) and f = (1+
            # exp(-delta(gdp/pop-phi))
            g_prime_f = (1 / nu) * (1 + np.exp(-delta *
                                               (gdp / pop - phi))) ** (1 / nu - 1)
            f_prime = - delta * (- d_pop * gdp) / (pop ** 2) * \
                np.exp(- delta * (gdp / pop - phi))
            w_prime = g_prime_f * f_prime
            # dr' = u'v + uv'
            d_base_death_rate = (- (dr_lower - dr_upper) * w_prime / w_squared)

            d_base_deathrate_d_temp[age_range] = d_base_death_rate

        return d_base_deathrate_d_temp

    def d_climate_death_rate_d_temp(self, year, d_base_death_rate):
        """
        for every columns of death rate dataframe create a dictionary that contains all derivatives wrt temp
        """
        nb_years = (self.year_end - self.year_start + 1)
        iyear = year - self.year_start
        idty = np.zeros(nb_years)
        idty[iyear] = 1
        param = self.dr_param_df
        param.index = param['param'].values

        d_climate_deathrate_d_temp = {}
        base_death_rate = self.base_death_rate_df.iloc[iyear, 1:]
        climate_death_rate = self.climate_death_rate_df.iloc[iyear, 1:]
        temp = self.temperature_df.loc[year, 'temp_atmo']
        cal_temp_increase = self.cal_temp_increase
        theta = self.theta
        add_death = self.climate_mortality_param_df
        add_death.index = add_death['param'].values

        for age_range in param['param'].values:
            # Param of death rate equation
            beta = add_death.loc[age_range, 'beta']
            d_climate_death_rate = beta * theta * idty / \
                cal_temp_increase * (temp / cal_temp_increase) ** (theta - 1)
            # (uv)' = u_prime_v * v_prime_u with u = base_deathrate v = climate_deathrate
            u_prime_v = d_base_death_rate[year][age_range] * \
                climate_death_rate[age_range] / base_death_rate[age_range]
            v_prime_u = d_climate_death_rate * base_death_rate[age_range]
            d_climate_deathrate_d_temp[age_range] = u_prime_v + v_prime_u

        return d_climate_deathrate_d_temp
