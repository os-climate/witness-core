'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/02-2023/11/03 Copyright 2023 Capgemini

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
from itertools import chain

import numpy as np
import pandas as pd
from pandas import DataFrame

from climateeconomics.glossarycore import GlossaryCore


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
        self.working_age_population_df = None
        self.birth_rate = None
        self.death_rate_df = None
        self.set_data(inputs)
        self.trillion = 1e12
        self.billion = 1e9
        self.million = 1e6

    def format_popu_init_df(self, input_df):
        age_col = list(GlossaryCore.PopulationStartDf["dataframe_descriptor"].keys())
        vals = input_df[age_col].values[0]
        out = pd.DataFrame({
            "age": age_col,
            "population": vals
        })
        return out

    def set_data(self, inputs):

        self.year_start = inputs[GlossaryCore.YearStart]
        self.year_end = inputs[GlossaryCore.YearEnd]
        self.time_step = inputs[GlossaryCore.TimeStep]
        self.pop_init_df = self.format_popu_init_df(inputs[GlossaryCore.PopulationStart])
        self.br_upper = inputs['birth_rate_upper']
        self.br_lower = inputs['birth_rate_lower']
        self.br_phi = inputs['birth_rate_phi']
        self.br_nu = inputs['birth_rate_nu']
        self.br_delta = inputs['birth_rate_delta']
        self.climate_mortality_param_df = deepcopy(inputs['climate_mortality_param_df'])
        # Pandemic parameters
        self.pandemic_mortality_df = deepcopy(inputs[GlossaryCore.PandemicParamDfValue]['mortality'])
        self.cal_temp_increase = inputs['calibration_temperature_increase']
        self.theta = inputs['theta']
        self.dr_param_df = deepcopy(inputs['death_rate_param'])
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
        self.diet_mortality_param_df = inputs['diet_mortality_param_df']
        self.kcal_pc_ref = inputs['kcal_pc_ref']
        self.theta_diet = inputs['theta_diet']
        self.activate_climate_effect_on_population = inputs['assumptions_dict']['activate_climate_effect_population']
        self.activate_pandemic_effects = inputs['assumptions_dict']['activate_pandemic_effects']
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
        self.pop_df_column = pop_column.copy()

        self.pop_df_column.append('total')

        # POPULATION
        # select all age except 100+
        pop_init = self.pop_init_df.iloc[:-1, 1]
        # Divide by 5
        pop_init = list(pop_init.divide(other=5))
        # Duplicate over age group
        pop_by_age = list(chain(*zip(*[pop_init for _ in range(5)])))
        # Add 100+ value of pop init
        pop_by_age.append(self.pop_init_df.iloc[-1, 1])

        self.total_pop = self.pop_init_df[GlossaryCore.PopulationValue].sum()
        self.population_dict = {GlossaryCore.YearStartDefault: np.array(
            pop_by_age + [self.total_pop])}

        column_list = self.age_list.copy()
        column_list.insert(0, GlossaryCore.Years)
        self.column_list = self.age_list.copy()
        # WORKING POULATION
        self.working_age_population_df = DataFrame(index=years_range,
                                                   columns=[GlossaryCore.Years, GlossaryCore.Population1570])
        self.working_age_population_df[GlossaryCore.Years] = years_range

        # BIRTH RATE
        # BASE => calculated from GDB and knowledge level
        self.birth_rate = DataFrame({GlossaryCore.Years: years_range,
                                     'knowledge': 0.,
                                     'birth_rate': 0.},
                                    index=years_range)

        # BIRTH NUMBER
        # BASE => calculated from GDB and knowledge level
        self.birth_df = DataFrame({GlossaryCore.Years: years_range,
                                   'knowledge': 0.,
                                   'birth_rate': 0.},
                                  index=years_range)

        # DEATH RATE - slices of 4 years age in column
        # BASE => calculated from GDP
        self.base_death_rate_df_dict = {}
        # CLIMATE => calculated from temperature increase
        self.climate_death_rate_df_dict = {}
        # DIET => calculated from kcal intake
        self.diet_death_rate_df_dict = {}
        # PANDEMIC => calculated from pandemic parameters
        self.pandemic_death_rate_df_dict = {}
        # TOTAL => sum of all effects
        self.death_rate_df_dict = {}

        # CONTAINER => dictionnary containing death rates dictionaties

        self.death_rate_dict = {
            'base': self.base_death_rate_df_dict,
            'climate': self.climate_death_rate_df_dict,
            'diet': self.diet_death_rate_df_dict,
            'pandemic': self.pandemic_death_rate_df_dict,
            'total': self.death_rate_df_dict}
        # DEATH NUMBER - one column per age
        # BASE => calculated from GDP
        init_dict = {GlossaryCore.Years: years_range}
        init_dict.update({col: 0 for col in self.pop_df_column})

        self.base_death_df = DataFrame(init_dict, index=years_range)
        # CLIMATE => calculated from temperature increase
        self.climate_death_df = DataFrame(init_dict.copy(), index=years_range)
        # TOTAL => sum of all effects
        self.death_df = DataFrame(init_dict.copy(), index=years_range)

        # CONTAINER => dictionnary containing death number
        self.death_dict = {}
        self.death_list_dict = {effect: []
                                for effect in self.death_rate_dict}
        # LIFE EXPECTANCY
        self.life_expectancy_df = DataFrame({GlossaryCore.Years: years_range,
                                             'life_expectancy': 0.}, index=years_range)

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
        gdp = self.economics_df.loc[year, GlossaryCore.OutputNetOfDamage] * self.trillion
        pop = self.total_pop
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
        gdp = self.economics_df.loc[year, GlossaryCore.OutputNetOfDamage] * self.trillion
        pop = self.total_pop
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
        gdp = self.economics_df.loc[year, GlossaryCore.OutputNetOfDamage] * self.trillion
        pop = self.total_pop
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
        gdp = self.economics_df.loc[year, GlossaryCore.OutputNetOfDamage] * self.trillion
        pop = self.total_pop
        temp = self.temperature_df.loc[year, GlossaryCore.TempAtmo]
        param = self.dr_param_df
        add_death = self.climate_mortality_param_df
        kcal_pc = self.calories_pc_df.loc[year, 'kcal_pc']
        kcal_pc_ref = self.kcal_pc_ref
        cal_temp_increase = self.cal_temp_increase
        theta = self.theta
        theta_diet = self.theta_diet
        # For all age range compute death rate
        death_rate = param['death_rate_upper'].values + (param['death_rate_lower'].values - param['death_rate_upper'].values) / (
            1 + np.exp(-param['death_rate_delta'].values * 
                       (gdp / pop - param['death_rate_phi'].values))) ** (1 / param['death_rate_nu'].values)
        # Add climate impact on death rate
        climate_death_rate = add_death['beta'].values * (temp / cal_temp_increase) ** theta
        if not self.activate_climate_effect_on_population:
            climate_death_rate *= 0

        # Add diet impact on death rate
        alpha_diet = self.diet_mortality_param_df['undernutrition'].values
        if kcal_pc >= kcal_pc_ref:
            alpha_diet = self.diet_mortality_param_df['overnutrition'].values

        if np.real(kcal_pc - kcal_pc_ref) >= 0:
            diet_death_rate = alpha_diet * (kcal_pc - kcal_pc_ref)/(theta_diet * kcal_pc_ref)
        else:
            diet_death_rate = alpha_diet * (kcal_pc_ref - kcal_pc)/(theta_diet * kcal_pc_ref)

        for i in range(len(death_rate)):
            if diet_death_rate[i] >= 1 - death_rate[i] * (1 + climate_death_rate[i]):
                diet_death_rate[i] = (1 - death_rate[i] * (1 + climate_death_rate[i]))/ (1 + np.exp(-diet_death_rate[i]))

        # Add pandemic impact on death rate
        pandemic_death_rate = self.pandemic_mortality_df.values
        if not self.activate_pandemic_effects:
            pandemic_death_rate *= 0

        # Fill the year key in each death rate dict
        # FIXME: why do we have [year] on the left and side and not also on the right hand side?
        self.base_death_rate_df_dict[year] = death_rate
        self.climate_death_rate_df_dict[year] = climate_death_rate * death_rate
        self.diet_death_rate_df_dict[year] = diet_death_rate
        self.pandemic_death_rate_df_dict[year] = pandemic_death_rate
        self.death_rate_df_dict[year] = death_rate * (1 + climate_death_rate) + diet_death_rate + pandemic_death_rate

    def compute_death_number(self, year):
        """Compute number of dead people per year
        input: population df
                death rate value per age range 
        output df of number of death 
        """

        # last value is always total
        pop_year = self.population_dict[year][:-1]
        total_deaths = np.zeros(len(pop_year))

        for effect in self.death_rate_dict:
            if effect != 'total':
                # get the death rate in the dict instead of the df at key year
                # except the 100+
                dr_year = self.death_rate_dict[effect][year][:-1]
                # Duplicate each element of the list 5 times so that we have a
                # death rate per age
                full_dr_death = np.repeat(dr_year, 5).tolist()
                # The last value is the 100+ that we do not want to repeat
                full_dr_death.append(
                    self.death_rate_dict[effect][year][-1])

                nb_death = pop_year * np.array(full_dr_death)
                total_deaths = total_deaths + nb_death
                self.death_list_dict[effect].append(nb_death)
            else:
                self.death_list_dict[effect].append(total_deaths)
        return total_deaths

    def compute_birth_number(self, year):
        '''Compute number of birth per year
        input: birth rate 
                population df 
        output df of number of birth per year
        '''
        # Sum population between 15 and 49year
        pop_1549 = sum(self.population_dict[year][15:50])
        nb_birth = self.birth_rate.loc[year, 'birth_rate'] * pop_1549
        self.birth_df.loc[year, 'number_of_birth'] = nb_birth

        return nb_birth

    def compute_population_next_year(self, year, total_death, nb_birth):
        if year > self.year_end:
            pass
        else:
            year_start = self.year_start
            pop_before = self.population_dict[year - 1][:-1]
            pop_before = pop_before - total_death

            # Add new born And +1 for each person alive
            population_list = [nb_birth] + list(pop_before[:-1])
            # Add not dead people over 100+
            old_not_dead = pop_before[-1]
            population_list[-1] += old_not_dead
            # compute the total
            self.total_pop = sum(population_list)
            # Add pop each age + total in the dict
            self.population_dict[year] = np.array(
                population_list + [self.total_pop])

    def compute_life_expectancy(self, year):
        """
        Compute life expectancy for a year
        life expectancy = sum(pop_i) with i the age 
        pop_0 = 1 
        pop_i = pop_i-1(1- death_rate_i) 
        """
        # dr_year = self.death_rate_df.iloc[year - self.year_start, 1:-1]
        # do not duplicate 100+
        dr_year = self.death_rate_df_dict[year][:-1]
        # Duplicate each element of the list 5 times so that we have a death
        # rate per age
        full_dr_death = list(np.repeat(dr_year, 5))
        full_dr_death.append(self.death_rate_df_dict[year][-1])

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

    def compute(self, in_dict) -> tuple[DataFrame, DataFrame, dict[str, DataFrame], DataFrame, dict, DataFrame, DataFrame]:
        """
        Compute all, returning tuple of:
            population df,
            birth_rate df,
            death_rate dict,
            birth df,
            death dict,
            life_expectancy df,
            working_age_pop df,
        """
        self.create_dataframe()
        year_range = self.years_range
        self.economics_df = in_dict[GlossaryCore.EconomicsDfValue]
        self.economics_df.index = self.economics_df[GlossaryCore.Years].values
        self.temperature_df = in_dict[GlossaryCore.TemperatureDfValue]
        self.temperature_df.index = self.temperature_df[GlossaryCore.Years].values
        self.calories_pc_df = in_dict[GlossaryCore.CaloriesPerCapitaValue]
        self.calories_pc_df.index = self.calories_pc_df[GlossaryCore.Years].values
        self.compute_knowledge()

        # Loop over year to compute population evolution. except last year
        for year in year_range:
            self.compute_birth_rate_v2(year)
            self.compute_death_rate_v2(year)
            total_death = self.compute_death_number(year)
            nb_birth = self.compute_birth_number(year)
            self.compute_population_next_year(year + 1, total_death, nb_birth)
            self.compute_life_expectancy(year)

        self.population_df = DataFrame.from_dict(
            self.population_dict, orient='index', columns=self.pop_df_column)

        self.population_df.insert(loc=0, column=GlossaryCore.Years,
                                  value=self.years_range)
        self.population_df = self.population_df.replace(
            [np.inf, -np.inf], np.nan)

        # Compute working age population between 15 and 70 years
        working_age_idx = [str(i) for i in np.arange(15, 71)]
        self.working_age_population_df[GlossaryCore.Population1570] = (
            self.population_df[working_age_idx]
            .sum(axis=1)
        )

        # reconstruction of the dataframes with the dictionaries
        self.climate_death_rate_df = DataFrame.from_dict(
            self.climate_death_rate_df_dict, orient='index', columns=self.column_list)
        self.diet_death_rate_df = DataFrame.from_dict(
            self.diet_death_rate_df_dict, orient='index', columns=self.column_list)
        self.pandemic_death_rate_df = DataFrame.from_dict(
            self.pandemic_death_rate_df_dict, orient='index', columns=self.column_list)
        self.base_death_rate_df = DataFrame.from_dict(
            self.base_death_rate_df_dict, orient='index', columns=self.column_list)
        self.death_rate_df = DataFrame.from_dict(
            self.death_rate_df_dict, orient='index', columns=self.column_list)

        # recontruction of the death rate_dict with dataframes instead of
        # ditionaries at the end of the year loop
        self.death_rate_dict = {'base': self.base_death_rate_df,
                                'climate': self.climate_death_rate_df,
                                'diet': self.diet_death_rate_df,
                                'pandemic': self.pandemic_death_rate_df,
                                'total': self.death_rate_df}
        # Calculation of cumulative deaths
        for effect in self.death_rate_dict:
            self.death_dict[effect] = DataFrame(
                self.death_list_dict[effect], index=self.years_range)
            self.death_dict[effect]['total'] = self.death_dict[effect].sum(
                axis=1, skipna=True)
            self.death_dict[effect]['cum_total'] = self.death_dict[effect]['total'].cumsum(
            )

        for effect in self.death_dict:
            self.death_dict[effect].fillna(0.0)
            self.death_rate_dict[effect].fillna(0.0)

        return self.population_df.fillna(0.0), self.birth_rate.fillna(0.0), self.death_rate_dict, \
            self.birth_df.fillna(
                0.0), self.death_dict, self.life_expectancy_df.fillna(0.0), self.working_age_population_df.fillna(0.0)

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
        d_pop_disabled_d_output = np.zeros((nb_years, nb_years))

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
                                                                d_death_rate_climate_d_output, d_pop_d_output,
                                                                activate_effect_on_population=self.activate_climate_effect_on_population)
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
        gdp = self.economics_df.loc[year, GlossaryCore.OutputNetOfDamage] * self.trillion
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
        d_f_gdp_d_output = -(br_lower - br_upper) * u_prime / u_squared

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

        br = self.birth_rate.loc[year, 'birth_rate']

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
        gdp = self.economics_df.loc[year, GlossaryCore.OutputNetOfDamage] * self.trillion
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

            d_death_rate_d_output_age = -\
                (br_lower - br_upper) * u_prime / u_squared
            d_deathrate_d_output[age_range] = d_death_rate_d_output_age
        return d_deathrate_d_output

    def d_climate_death_rate_d_output(self, year, d_base_death_rate):
        """
        for every columns of climate death rate dataframe create a dictionary that contains all derivatives wrt output
        """
        nb_years = (self.year_end - self.year_start + 1)
        iyear = year - self.year_start
        idty = np.zeros(nb_years)
        idty[iyear] = 1
        param = self.dr_param_df
        param.index = param['param'].values

        d_climate_deathrate_d_output = {}
        climate_death_rate = self.climate_death_rate_df.iloc[iyear, :]
        temp = self.temperature_df.loc[year, GlossaryCore.TempAtmo]
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
                          dict_d_population_d_output, activate_effect_on_population=True):
        """
        Compute derivative of each column of death df wrt output and returns a dictionary
        has been initially developed for d_death_d_output which can take into account effect of climate on population
        was then used for d_death_d_temp and d_death_d_k_cal. In order to activate the effect of diet on death rate,
        this activate_effect_on_population must be set to true

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
        dr_year = self.death_rate_dict['total'].iloc[iyear, :-1]
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
            climate_list_d_dr_d_out_i = climate_list_d_dr_d_out[i] if activate_effect_on_population else 0
            d_death[ages[i]] = list_d_pop_d_out[ages[i]] * full_dr_death[i] + \
                (climate_list_d_dr_d_out_i +
                 base_list_d_dr_d_out[i]) * pop_year.values[i]

        return d_death

    def d_poptotal_generic(self, year, d_pop, d_death, d_birth, d_pop_1549,
                           d_total_pop, d_working_pop):
        """
        Compute derivative of column total of pop df wrt output
        """
        ## refactored to improve performances (could still be improved)
        if year + 1 <= self.year_end:
            iyear = year - self.year_start
            number_of_values = (self.year_end - self.year_start + 1)

            age_list = self.population_df.columns[1:-1]
            range_age_1549 = np.arange(15, 49 + 1)  # 15 to 49
            range_age_1570 = np.arange(15, 70 + 1)  # 15 to 70

            sum_pop_1549 = np.zeros(number_of_values)
            sum_pop_1570 = np.zeros(number_of_values)

            d_pop_d_age_prev = {}
            d_pop_d_y = {}

            # at age = 0 pop = nb_birth => d_pop = d_nb_birth
            age_list_0 = age_list[0]
            d_pop_d_y[age_list_0] = d_birth[iyear]
            d_pop_d_age_prev[age_list_0] = (
                    d_pop.get(year, {}).get(age_list_0, 0) - d_death[year][age_list_0]  # .get(age, 0)
            )

            sum_tot_pop = np.zeros(number_of_values)
            sum_tot_pop += d_pop_d_y[age_list_0]
            for i in range(1, len(age_list)):
                age = age_list[i]
                # Compute population of previous year: population - nb_death =>
                # Derivative = d_pop - d_nb_death
                d_pop_d_age_prev[age] = (
                        d_pop.get(year, {}).get(age, 0) - d_death[year][age]  # .get(age, 0)
                )
                # at year age between 1 and 100+: = pop_before =>
                # derivative = d_pop_before
                d_pop_d_y[age] = d_pop_d_age_prev[age_list[i - 1]]
                if i in range_age_1549:
                    sum_pop_1549 += d_pop_d_y[age]
                if i in range_age_1570:
                    sum_pop_1570 += d_pop_d_y[age]

            sum_tot_pop += np.sum(list(d_pop_d_y[age] for age in age_list[1::]), axis=0)

            # Add old not dead at year before at 100+ this year
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
        if not self.activate_climate_effect_on_population:
            return d_pop_tot_d_temp, d_working_pop_d_temp
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
                    year, d_base_death_rate, d_climate_death_rate, d_pop_d_temp,
                    activate_effect_on_population=self.activate_climate_effect_on_population)
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
        gdp = self.economics_df.loc[year, GlossaryCore.OutputNetOfDamage] * self.trillion
        d_pop = d_pop_tot_d_temp[iyear]
        idty = np.zeros(nb_years)
        idty[iyear] = 1
        # f gdp = br_upper + (self.br_lower - self.br_upper)/ u
        # -> fprime = (self.br_lower - self.br_upper)*u_prime/u_squared
        u_squared = ((1 + np.exp(-delta * (gdp / pop - phi)))
                     ** (1 / nu)) ** 2
        # u = g(f(x))  #u = g(f(x)) with g = f**(1/nu) and f = 1+
        # exp(-delta(gdp/pop-phi))
        g_prime_f = (1 / nu) * (1 + np.exp(-delta * 
                                           (gdp / pop - phi))) ** (1 / nu - 1)
        f_prime = -delta * (-gdp * d_pop) / (pop ** 2) * \
            np.exp(-delta * (gdp / pop - phi))
        u_prime = g_prime_f * f_prime
        d_f_gdp_d_temp = -(br_lower - br_upper) * u_prime / u_squared

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
        gdp = self.economics_df.loc[year, GlossaryCore.OutputNetOfDamage] * self.trillion
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
            f_prime = -delta * (-d_pop * gdp) / (pop ** 2) * \
                np.exp(-delta * (gdp / pop - phi))
            w_prime = g_prime_f * f_prime
            # dr' = u'v + uv'
            d_base_death_rate = (-(dr_lower - dr_upper) * w_prime / w_squared)

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
        base_death_rate = self.base_death_rate_df.iloc[iyear, :]
        climate_death_rate = self.climate_death_rate_df.iloc[iyear, :]
        temp = self.temperature_df.loc[year, GlossaryCore.TempAtmo]
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

    # WRT KCAL PC
    def compute_d_pop_d_kcal_pc(self):
        """ Compute the derivative of population wrt calories per capita
        """
        nb_years = (self.year_end - self.year_start + 1)

        # derivative of population of the current year for each age
        d_pop_d_kcal_pc = {}
        d_working_pop_d_kcal_pc = np.zeros((nb_years, nb_years))
        d_pop_tot_d_kcal_pc = np.zeros((nb_years, nb_years))
        d_birthrate_d_kcal_pc = np.zeros((nb_years, nb_years))
        d_birth_d_kcal_pc = np.zeros((nb_years, nb_years))
        d_base_death_rate = {}
        d_diet_death_rate = {}
        d_death_d_kcal_pc = {}
        d_pop_1549_d_kcal_pc = np.zeros((nb_years, nb_years))

        for year in self.years_range:
            if year == self.year_end:
                pass
            else:
                d_birthrate_d_kcal_pc += self.d_birthrate_d_kcal_pc(
                    year, d_pop_tot_d_kcal_pc)
                d_birth_d_kcal_pc += self.d_birth_d_generic(
                    year, d_birthrate_d_kcal_pc, d_pop_1549_d_kcal_pc)
                d_base_death_rate[year] = self.d_base_death_rate_d_kcal_pc(
                    year, d_pop_tot_d_kcal_pc)
                d_diet_death_rate[year] = self.d_diet_death_rate_d_kcal_pc(year, d_base_death_rate)
                d_death_d_kcal_pc[year] = self.d_death_d_generic(
                    year, d_base_death_rate, d_diet_death_rate, d_pop_d_kcal_pc,
                    activate_effect_on_population=True) # must activate d_pop_d_kcal_pc effect on pop of
                d_pop_d_kcal_pc, d_pop_1549_d_kcal_pc, d_pop_tot_d_kcal_pc, d_working_pop_d_kcal_pc = self.d_poptotal_generic(year, d_pop_d_kcal_pc,
                                                                                                                  d_death_d_kcal_pc,
                                                                                                                  d_birth_d_kcal_pc,
                                                                                                                  d_pop_1549_d_kcal_pc,
                                                                                                                  d_pop_tot_d_kcal_pc,
                                                                                                                  d_working_pop_d_kcal_pc)
        return d_pop_tot_d_kcal_pc, d_working_pop_d_kcal_pc

    def d_birthrate_d_kcal_pc(self, year, d_pop_tot_d_kcal_pc):
        """ Compute the derivative of birth rate wrt kcal_pc
        f_knowledge = self.cst_br_k + self.alpha_br_k * (1- knowledge/100)**self.beta_br_k
        f_gdp = self.br_upper + (self.br_lower - self.br_upper)/(1+np.exp(-self.br_delta*(gdp/pop -self.br_phi)))**(1/self.br_nu)
        birth_rate = self.share_know * f_knowledge + (1-self.share_know) * f_gdp
        """
        nb_years = (self.year_end - self.year_start + 1)
        d_birthrate_d_kcal_pc = np.zeros((nb_years, nb_years))
        iyear = year - self.year_start
        # Param of birth rate equation
        delta = self.br_delta
        phi = self.br_phi
        br_upper = self.br_upper
        br_lower = self.br_lower
        nu = self.br_nu
        pop = self.population_df.loc[year, 'total']
        gdp = self.economics_df.loc[year, GlossaryCore.OutputNetOfDamage] * self.trillion
        d_pop = d_pop_tot_d_kcal_pc[iyear]
        idty = np.zeros(nb_years)
        idty[iyear] = 1
        # f gdp = br_upper + (self.br_lower - self.br_upper)/ u
        # -> fprime = (self.br_lower - self.br_upper)*u_prime/u_squared
        u_squared = ((1 + np.exp(-delta * (gdp / pop - phi)))
                     ** (1 / nu)) ** 2
        # u = g(f(x))  #u = g(f(x)) with g = f**(1/nu) and f = 1+
        # exp(-delta(gdp/pop-phi))
        g_prime_f = (1 / nu) * (1 + np.exp(-delta * 
                                           (gdp / pop - phi))) ** (1 / nu - 1)
        f_prime = -delta * (-gdp * d_pop) / (pop ** 2) * \
            np.exp(-delta * (gdp / pop - phi))
        u_prime = g_prime_f * f_prime
        d_f_gdp_d_kcal_pc = -(br_lower - br_upper) * u_prime / u_squared

        d_f_gdp_d_net_kcal_pc = d_f_gdp_d_kcal_pc
        # and lastly multiply by (1-share now)
        d_birthrate_d_kcal_pc[iyear] = (1 - self.share_know) * d_f_gdp_d_net_kcal_pc

        return d_birthrate_d_kcal_pc

    def d_base_death_rate_d_kcal_pc(self, year, d_pop_tot_d_kcal_pc):
        """
        for every columns of death rate dataframe create a dictionary that contains all derivatives wrt kcal_pc
        """
        nb_years = (self.year_end - self.year_start + 1)
        iyear = year - self.year_start
        idty = np.zeros(nb_years)
        idty[iyear] = 1

        param = self.dr_param_df
        param.index = param['param'].values

        d_base_deathrate_d_kcal_pc = {}
        pop = self.population_df.loc[year, 'total']
        gdp = self.economics_df.loc[year, GlossaryCore.OutputNetOfDamage] * self.trillion
        temp = self.temperature_df.loc[year, GlossaryCore.TempAtmo]
        add_death = self.climate_mortality_param_df
        add_death.index = add_death['param'].values
        cal_temp_increase = self.cal_temp_increase
        theta = self.theta
        add_death = self.climate_mortality_param_df
        d_pop = d_pop_tot_d_kcal_pc[iyear]

        for age_range in param['param'].values:
            # Param of death rate equation
            delta = param.loc[age_range, 'death_rate_delta']
            beta = add_death.loc[age_range, 'beta']
            phi = param.loc[age_range, 'death_rate_phi']
            dr_upper = param.loc[age_range, 'death_rate_upper']
            dr_lower = param.loc[age_range, 'death_rate_lower']
            nu = param.loc[age_range, 'death_rate_nu']

            # Value on the diagonal. death rate t depends on kcal_pc t
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
            f_prime = -delta * (-d_pop * gdp) / (pop ** 2) * \
                np.exp(-delta * (gdp / pop - phi))
            w_prime = g_prime_f * f_prime
            # dr' = u'v + uv'
            d_base_death_rate = (-(dr_lower - dr_upper) * w_prime / w_squared)

            d_base_deathrate_d_kcal_pc[age_range] = d_base_death_rate + d_base_death_rate * beta * (temp / cal_temp_increase) ** theta

        return d_base_deathrate_d_kcal_pc

    def d_diet_death_rate_d_kcal_pc(self, year, d_base_death_rate):
        """
        for every columns of death rate dataframe create a dictionary that contains all derivatives wrt kcal_pc
        """
        nb_years = (self.year_end - self.year_start + 1)
        iyear = year - self.year_start
        idty = np.zeros(nb_years)
        idty[iyear] = 1
        param = self.dr_param_df
        param.index = param['param'].values

        d_diet_deathrate_d_kcal_pc = {}
        base_death_rate = self.base_death_rate_df.iloc[iyear, :]
        climate_death_rate = self.climate_death_rate_df.iloc[iyear, :]
        add_death = self.climate_mortality_param_df
        add_death.index = add_death['param'].values
        diet_death_param = self.diet_mortality_param_df
        diet_death_param.index = diet_death_param['param'].values
        kcal_pc = self.calories_pc_df.loc[year, 'kcal_pc']
        kcal_pc_ref = self.kcal_pc_ref
        theta_diet = self.theta_diet
        risk_type = 'undernutrition'
        if kcal_pc >= kcal_pc_ref:
            risk_type = 'overnutrition'
        alpha_diet = diet_death_param[risk_type]
        if np.real(kcal_pc - kcal_pc_ref) >= 0:
            diet_death_rate = alpha_diet * (kcal_pc - kcal_pc_ref)/(theta_diet * kcal_pc_ref)
        else:
            diet_death_rate = alpha_diet * (kcal_pc_ref - kcal_pc)/(theta_diet * kcal_pc_ref)

        for age_range in param['param'].values:
            # Param of death rate equation
            alpha_diet = diet_death_param.loc[age_range, risk_type]
            if np.real(kcal_pc - kcal_pc_ref) >= 0:
                d_diet_deathrate_d_kcal_pc[age_range] = alpha_diet * idty / (theta_diet * kcal_pc_ref)
            else:
                d_diet_deathrate_d_kcal_pc[age_range] = - alpha_diet * idty / (theta_diet * kcal_pc_ref)

            if diet_death_rate[age_range] >= 1 - base_death_rate[age_range] - climate_death_rate[age_range]:
                # self.diet_death_rate_df_dict[year][age_range] = (1 - death_rate[i] * (1 + climate_death_rate[age_range]))/ (1 + np.exp(-self.diet_death_rate_df_dict[year][age_range]))
                # self.diet_death_rate_df_dict[year][age_range] = u / v
                # d_diet_death_rate_df_dict[year][age_range] = (du*v - dv*u)/v2
                u = 1 - base_death_rate[age_range]  - climate_death_rate[age_range]
                u_prime = - d_base_death_rate[year][age_range]
                v = 1 + np.exp(-diet_death_rate[age_range])
                v_prime = -d_diet_deathrate_d_kcal_pc[age_range] * np.exp(-diet_death_rate[age_range])
                d_diet_deathrate_d_kcal_pc[age_range] = (u_prime*v - v_prime * u)/(v**2)

        return d_diet_deathrate_d_kcal_pc
