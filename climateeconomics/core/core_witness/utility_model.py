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

from climateeconomics.core.core_witness.utility_tools import (
    compute_utility_objective_bis,
    compute_utility_quantities_bis,
)
from climateeconomics.glossarycore import GlossaryCore


class UtilityModel():
    '''
    Used to compute population welfare and utility
    '''

    def __init__(self, param):
        '''
        Constructor
        '''
        self.decreasing_gpd_obj = None
        self.net_gdp_growth_rate_obj = None
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
        
        self.shift_scurve = self.param['shift_scurve']
        self.strech_scurve = self.param['strech_scurve']

        self.conso_elasticity = self.param['conso_elasticity']  # elasmu
        self.init_rate_time_pref = self.param['init_rate_time_pref']  # prstp
        self.init_discounted_utility = self.param['init_discounted_utility']

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        years_range = np.arange(self.year_start, self.year_end + 1)
        self.years_range = years_range
        self.n_years = len(self.years_range)

    def compute(self, economics_df, energy_mean_price, population_df, multiply_by_pop: bool):
        """compute"""

        self.economics_df = economics_df
        energy_price = energy_mean_price[GlossaryCore.EnergyPriceValue].values

        population = population_df[GlossaryCore.PopulationValue].values
        consumption_pc = economics_df[GlossaryCore.PerCapitaConsumption].values

        utility_quantities = compute_utility_quantities_bis(self.years_range, consumption_pc, energy_price, population,
                                                        self.init_rate_time_pref, self.shift_scurve,
                                                        self.strech_scurve)

        self.utility_df = pd.DataFrame({GlossaryCore.Years: self.years_range} | utility_quantities)
        self.discounted_utility_quantity_objective = np.array([compute_utility_objective_bis(self.years_range, consumption_pc, energy_price, population,
                                                        self.init_rate_time_pref, self.shift_scurve,
                                                        self.strech_scurve, multiply_by_pop)])

        self.compute_decreasing_gdp_obj()
        self.compute_net_gdp_growth_rate_obj()


    ######### GRADIENTS ########
    def compute_decreasing_gdp_obj(self):
        """
        decreasing net gdp obj =   Sum_i [min(Qi+1/Qi, 1) - 1] / nb_years

        Note: this objective is self normalized to [0,1], no need for reference.
        It should be minimized and not maximized !
        :return:
        :rtype:
        """
        output_net_of_damage = self.economics_df[GlossaryCore.OutputNetOfDamage].values
        increments = list(output_net_of_damage[1:]/output_net_of_damage[:-1])
        increments.append(0)
        increments = np.array(increments)

        increments[increments >= 1] = 1.
        increments -= 1
        increments[-1] = 0

        self.decreasing_gpd_obj = - np.array([np.mean(increments)])

    def d_decreasing_gdp_obj(self):
        output_net_of_damage = self.economics_df[GlossaryCore.OutputNetOfDamage].values
        output_shift = list(output_net_of_damage[1:])
        output_shift.append(0)
        output_shift = np.array(output_shift)

        increments = list(output_net_of_damage[1:] / output_net_of_damage[:-1])
        increments.append(0)
        increments = np.array(increments)

        a = list(- output_shift / output_net_of_damage**2)
        derivative = np.diag(a) + np.diag(1/output_net_of_damage[:-1], k=1)

        derivative[increments > 1] = 0.
        for i, incr in enumerate(increments):
            if incr == 1:
                derivative[i, i+1] = 0.
        derivative = -np.mean(derivative, axis=0)

        return derivative

    def compute_net_gdp_growth_rate_obj(self):
        """
        decreasing net gdp obj =   Sum_i [Qi+1/Qi, 1 - 1] / nb_years

        Note: this objective is self normalized to [0,1], no need for reference.
        It should be minimized and not maximized !
        :return:
        :rtype:
        """
        output_net_of_damage = self.economics_df[GlossaryCore.OutputNetOfDamage].values
        increments = list(output_net_of_damage[1:]/output_net_of_damage[:-1])
        increments.append(0)
        increments = np.array(increments)

        increments -= 1
        increments[-1] = 0

        self.net_gdp_growth_rate_obj = - np.array([np.mean(increments)])

    def d_net_gdp_growth_rate_obj(self):
        output_net_of_damage = self.economics_df[GlossaryCore.OutputNetOfDamage].values
        output_shift = list(output_net_of_damage[1:])
        output_shift.append(0)
        output_shift = np.array(output_shift)

        increments = list(output_net_of_damage[1:] / output_net_of_damage[:-1])
        increments.append(0)
        increments = np.array(increments)

        a = list(- output_shift / output_net_of_damage**2)
        derivative = np.diag(a) + np.diag(1/output_net_of_damage[:-1], k=1)

        for i, incr in enumerate(increments):
            if incr == 1:
                derivative[i, i+1] = 0.
        derivative = -np.mean(derivative, axis=0)

        return derivative
