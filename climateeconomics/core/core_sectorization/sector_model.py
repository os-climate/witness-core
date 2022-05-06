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
        #self.init_gross_output = inputs_dict['init_gross_output']
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
        self.ref_emax_enet_constraint = inputs_dict['ref_emax_enet_constraint']
        
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
        self.investment_df = inputs['sector_investment']
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
            investment = self.investment_df.loc[year - self.time_step, 'investment']
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
    
    ### CONSTRAINTS ###
    def compute_emax_enet_constraint(self):
        """ Equation for Emax constraint 
        """
        e_max = self.capital_df['e_max'].values
        energy = self.energy_production['Total production'].values
        self.emax_enet_constraint = - \
            (energy - e_max * self.max_capital_utilisation_ratio) / self.ref_emax_enet_constraint
    
    #RUN
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
        self.compute_emax_enet_constraint()

        return self.production_df, self.capital_df, self.productivity_df, self.emax_enet_constraint
    
    ### GRADIENTS ###

    def compute_doutput_dworkforce(self):
        """ Gradient for output output wrt workforce
        output = productivity * (alpha * capital_u**gamma + (1-alpha)* (working_pop)**gamma)**(1/gamma) 
        """
        years = self.years_range
        nb_years = len(years)
        alpha = self.output_alpha
        gamma = self.output_gamma
        doutput = np.identity(nb_years)
        working_pop = self.workforce_df['workforce'].values
        capital_u = self.capital_df['usable_capital'].values
        productivity = self.productivity_df['productivity'].values
        # output = f(g(x)) with f = productivity*g**(1/gamma) a,d g= alpha * capital_u**gamma + (1-alpha)* (working_pop)**gamma
        # f'(g) = productivity*(1/gamma)*g**(1/gamma -1)
        # g'(workingpop) = (1-alpha)*gamma*workingpop**(gamma-1)
        # f'(g(x)) = f'(g)*g'(x)
        # first line stays at zero since derivatives of initial values are zero
        g = alpha * capital_u**gamma + (1 - alpha) * (working_pop)**gamma
        g_prime = (1 - alpha) * gamma * working_pop**(gamma - 1)
        f_prime = productivity * (1 / gamma) * g * g_prime
        doutput *= f_prime
        return doutput
    
    def dusablecapital_denergy(self):
        """ Gradient of usable capital wrt energy 
        usable_capital = capital * (energy / e_max)  
        """
        #derivative: capital/e_max
        nb_years = self.nb_years
        # Inputs
        capital = self.capital_df['capital'].values
        e_max = self.capital_df['e_max'].values
        dusablecapital_denergy = np.identity(nb_years)
        dusablecapital_denergy *= capital / e_max
        return dusablecapital_denergy
    
    def doutput_denergy(self, dcapitalu_denergy):
        years = self.years_range
        nb_years = len(years)
        alpha = self.output_alpha
        gamma = self.output_gamma
        doutput_dcap = np.identity(nb_years)
        working_pop = self.workforce_df['workforce'].values
        capital_u = self.capital_df['usable_capital'].values
        productivity = self.productivity_df['productivity'].values
        # Derivative of output wrt capital
        # output = f(g(x)) with f = productivity*g**(1/gamma) a,d g= alpha * capital_u**gamma + (1-alpha)* (working_pop)**gamma
        # f'(g) = productivity*(1/gamma)*g**(1/gamma -1)
        # g'(capital) = alpha*gamma*capital**(gamma-1)
        # f'(g(x)) = f'(g)*g'(x)
        g = alpha * capital_u**gamma + (1 - alpha) * (working_pop)**gamma
        g_prime = alpha * gamma * capital_u**(gamma - 1)
        f_prime = productivity * (1 / gamma) * g * g_prime
        doutput_dcap *= f_prime
        # Then doutput = doutput_d_prod * dproductivity
        doutput = np.dot(dcapitalu_denergy, doutput_dcap)
        return doutput
    
    def dproductivity_ddamage(self):
        """gradient for productivity for damage_df
        Args:
            output: gradient
        """
        years = np.arange(self.year_start,
                          self.year_end + 1, self.time_step)
        nb_years = len(years)
        p_productivity_gr = self.productivity_df['productivity_growth_rate'].values
        p_productivity = self.productivity_df['productivity'].values

        # derivative matrix initialization
        d_productivity = np.zeros((nb_years, nb_years))
        if self.damage_to_productivity == True:

            # first line stays at zero since derivatives of initial values are
            # zero
            for i in range(1, nb_years):
                d_productivity[i, i] = (1 - self.frac_damage_prod * self.damefrac.at[years[i], 'damage_frac_output']) * \
                                    d_productivity[i - 1, i] / (1 - p_productivity_gr[i - 1]) - self.frac_damage_prod * \
                                    p_productivity[i - 1] / (1 - p_productivity_gr[i - 1])
                for j in range(1, i):
                    d_productivity[i, j] = (1 - self.frac_damage_prod * self.damefrac.at[years[i], 'damage_frac_output']) * \
                                            d_productivity[i - 1, j] / (1 - p_productivity_gr[i - 1] )

        return d_productivity
    
    def doutput_ddamage(self, dproductivity):
        years = self.years_range
        nb_years = len(years)
        alpha = self.output_alpha
        gamma = self.output_gamma
        doutput_dprod = np.identity(nb_years)
        working_pop = self.workforce_df['workforce'].values
        capital_u = self.capital_df['usable_capital'].values
        # Derivative of output wrt productivity
        doutput_dprod *= (alpha * capital_u**gamma + (1 - alpha)
                          * (working_pop)**gamma)**(1 / gamma)
        # Then doutput = doutput_d_prod * dproductivity
        doutput = np.dot(doutput_dprod, dproductivity)
        return doutput
        
    def dcapital_dinvest(self):
        """ Compute derivative of capital wrt investments. 
        """
        nb_years = self.nb_years
        #capital depends on invest from year before. diagonal k-1
        dcapital = np.eye(nb_years, k=-1)
        for i in range(0, nb_years-1):
            for j in range(0, i + 1):
                dcapital[i + 1, j] += dcapital[i, j] * (1 - self.depreciation_capital)  

        return dcapital
    
    def demaxconstraint(self, dcapital):
        """ Compute derivative of e_max and emax constraint using derivative of capital. 
        For all inputs that impacts e_max through capital 
        """
        #e_max = capital*1e3/ (capital_utilisation_ratio * energy_efficiency)
        energy_efficiency = self.capital_df['energy_efficiency'].values
        demax = np.identity(self.nb_years)
        demax *= 1e3 / (self.capital_utilisation_ratio * energy_efficiency)
        demax = np.dot(demax, dcapital)
        demaxconstraint_demax = demax * self. max_capital_utilisation_ratio / self.ref_emax_enet_constraint
        return demaxconstraint_demax
       

