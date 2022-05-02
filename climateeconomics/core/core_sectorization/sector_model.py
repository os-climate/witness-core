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
from os.path import join, dirname
from gemseo.third_party.prettytable.prettytable import NONE


class SectorModel():
    """
    Sector model
    General implementation of sector model 
    """

    #Units conversion
    conversion_factor=1.0

    def __init__(self):
        '''
        Constructor
        '''
        self.productivity_df = None
        self.capital_df = None
        self.production_df = None
        self.workforce_df = None
        
    def configure_parameters(self, inputs_dict):
        '''
        Configure with inputs_dict from the discipline
        '''

        self.year_start = inputs_dict['year_start']  # year start
        self.year_end = inputs_dict['year_end']  # year end
        self.time_step = inputs_dict['time_step']
        self.years_range = np.arange(self.year_start,self.year_end + 1,self.time_step)
        self.nb_years = len(self.years_range)

        self.productivity_start = inputs_dict['productivity_start']
        self.init_gross_output = inputs_dict['init_gross_output']
        self.capital_start = inputs_dict['capital_start']
        self.productivity_gr_start = inputs_dict['productivity_gr_start']
        self.decline_rate_tfp = inputs_dict['decline_rate_tfp']
        self.depreciation_capital = inputs_dict['depreciation_capital']
        self.frac_damage_prod = inputs_dict['frac_damage_prod']
        self.damage_to_productivity = inputs_dict['damage_to_productivity']
        self.init_output_growth = inputs_dict['init_output_growth']
        self.output_alpha = inputs_dict['output_alpha']
        self.output_gamma = inputs_dict['output_gamma']
        self.energy_eff_k = inputs_dict['energy_eff_k']
        self.energy_eff_cst = inputs_dict['energy_eff_cst']
        self.energy_eff_xzero = inputs_dict['energy_eff_xzero']
        self.energy_eff_max = inputs_dict['energy_eff_max']
        self.capital_utilisation_ratio = inputs_dict['capital_utilisation_ratio']
        self.max_capital_utilisation_ratio = inputs_dict['max_capital_utilisation_ratio']
        self.scaling_factor_energy_production = inputs_dict['scaling_factor_energy_production']
        
        self.init_dataframes()

    def init_dataframes(self):
        '''
        Init dataframes with years
        '''
        self.years = np.arange(self.year_start, self.year_end + 1)
        default_index = self.years
        self.capital_df = pd.DataFrame(index=default_index,columns=['years','energy_efficiency', 'e_max', 'capital', 'usable_capital'])
        self.production_df = pd.DataFrame(index=default_index,columns=['years','output'])
        self.productivity_df = pd.DataFrame(index=default_index,columns=['years','productivity_growth_rate', 'productivity'])
        self.production_df['years'] = self.years
        self.capital_df['years'] = self.years
        self.productivity_df['years'] = self.years
        self.capital_df.loc[self.year_start, 'capital'] = self.capital_start
    
    def set_coupling_inputs(self, inputs):
        """
        Set couplings inputs with right index, scaling... 
        """
        self.investment_df = inputs['investment']
        self.investment_df.index = self.investment_df['years'].values
        #scale energy production
        self.energy_production = inputs['energy_production'].copy(deep=True)
        self.energy_production['Total production'] *= self.scaling_factor_energy_production
        self.energy_production.index = self.energy_production['years'].values
        self.workforce_df = inputs['workforce_df']
        self.workforce_df.index = self.workforce_df['years'].values
        self.damage_df = inputs['damage_df']
        self.damage_df.index = self.damage_df['years'].values

    def compute_productivity_growthrate(self):
        '''
        A_g, Growth rate of total factor productivity without damage 
        Returns:
            :returns: A_g(0) * exp(-Δ_a * (t-1))
        '''
        t = np.arange(0, len(self.years))
        productivity_gr = self.productivity_gr_start * np.exp(-self.decline_rate_tfp * t)
        productivity_gr /= 5  
        self.productivity_df['productivity_growth_rate'] = productivity_gr
        return productivity_gr

    def compute_productivity(self, year):
        '''
        productivity
        if damage_to_productivity= True add damage to the the productivity
        if  not: productivity evolves independently from other variables (except productivity growthrate)
        '''
        damage_to_productivity = self.damage_to_productivity
        damefrac = self.damage_df.at[year, 'damage_frac_output']
        #For year_start put initial value 
        if year == self.year_start: 
            productivity =  self.productivity_start  
        #for other years: two ways to compute:  
        elif damage_to_productivity == True:
            p_productivity = self.productivity_df.at[year -self.time_step, 'productivity']
            p_productivity_gr = self.productivity_df.at[year - self.time_step, 'productivity_growth_rate']
            #damage = 1-damefrac
            productivity = (1 - self.frac_damage_prod * damefrac) *(p_productivity / (1 - p_productivity_gr))
        else:
            p_productivity = self.productivity_df.at[year -self.time_step, 'productivity']
            p_productivity_gr = self.productivity_df.at[year - self.time_step, 'productivity_growth_rate']
            productivity = p_productivity /(1 - p_productivity_gr)
        # we divide the productivity growth rate by 5/time_step because of change in time_step (as advised in Traeger, 2013)
        self.productivity_df.loc[year, 'productivity'] = productivity
        return productivity

    def compute_capital(self, year):
        """
        K(t), Capital for time period, trillions $USD
        Args:
            :param capital: capital
            :param depreciation: depreciation rate
            :param investment: investment
            K(t) = K(t-1)*(1-depre_rate) + investment(t-1)
        """
        if year > self.year_end:
            pass
        else: 
            # Capital 
            investment = self.investment_df.at[year - self.time_step, 'investment']
            capital = self.capital_df.at[year - self.time_step, 'capital']
            capital_a = capital * (1 - self.depreciation_capital) + investment
            self.capital_df.loc[year, 'capital'] = capital_a
                                  
            return capital_a

    def compute_emax(self, year):
        """E_max is the maximum energy capital can use to produce output
        E_max = K/(capital_utilisation_ratio*energy_efficiency(year)
        energy_efficiency = 1+ max/(1+exp(-k(x-x0)))
        energy_efficiency is a logistic function because it represent technological progress
        """
        k = self.energy_eff_k
        cst = self.energy_eff_cst
        xo = self.energy_eff_xzero
        capital_utilisation_ratio = self.capital_utilisation_ratio
        max_e = self.energy_eff_max
        # Convert capital in billion: to get same order of magnitude (1e6) as energy 
        capital = self.capital_df.loc[year, 'capital'] * 1e3
        # compute energy_efficiency
        energy_efficiency = cst + max_e / (1 + np.exp(-k * (year - xo)))
        if year == 2020: 
            energy_efficiency = 2.7565
        # Then compute e_max
        e_max = capital / (capital_utilisation_ratio * energy_efficiency)

        self.capital_df.loc[year,'energy_efficiency'] = energy_efficiency
        self.capital_df.loc[year, 'e_max'] = e_max

    def compute_usable_capital(self, year):
        """  Usable capital is the part of the capital stock that can be used in the production process. 
        To be usable the capital needs enough energy.
        K_u = K*(E/E_max) 
        E is energy in Twh and K is capital in trill dollars constant 2020
        Output: usable capital in trill dollars constant 2020
        """
        capital = self.capital_df.loc[year, 'capital']
        energy = self.energy_production.at[year, 'Total production']
        e_max = self.capital_df.loc[year, 'e_max']
        # compute usable capital
        usable_capital = capital * (energy / e_max)
        self.capital_df.loc[year, 'usable_capital'] = usable_capital
        return usable_capital

    def compute_gross_output(self, year):
        """ Compute the gdp 
        inputs: usable capital by year in trill $ , working population by year in million of people,
             productivity by year (no unit), alpha (between 0 and 1) 
        output: gdp in trillion dollars
        """
        alpha = self.output_alpha
        gamma = self.output_gamma
        productivity = self.productivity_df.loc[year, 'productivity']
        working_pop = self.workforce_df.loc[year, 'workforce']
        capital_u = self.capital_df.loc[year, 'usable_capital']
        # If gamma == 1/2 use sqrt but same formula
        if gamma == 1 / 2:
            output = productivity * \
                (alpha * np.sqrt(capital_u) + (1 - alpha) * np.sqrt(working_pop))**2
        else:
            output = productivity * \
                (alpha * capital_u**gamma + (1 - alpha)* (working_pop)**gamma)**(1 / gamma)
        self.production_df.loc[year, 'output'] = output

        return output
    
    def compute(self, inputs):
        """
        Compute all models for year range
        """
        self.init_dataframes()
        self.inputs = inputs
        self.set_coupling_inputs(inputs)
        self.compute_productivity_growthrate()
        # iterate over years
        for year in self.years_range:
            self.compute_productivity(year)
            self.compute_emax(year)
            self.compute_usable_capital(year)
            self.compute_gross_output(year)
            # capital t+1 :
            self.compute_capital(year+1)
        self.production_df = self.production_df.fillna(0.0)
        self.capital_df = self.capital_df.fillna(0.0)
        self.productivity_df = self.productivity_df.fillna(0.0)
        
        return self.production_df, self.capital_df, self.productivity_df


