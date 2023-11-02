'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/29-2023/11/02 Copyright 2023 Capgemini

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

from climateeconomics.glossarycore import GlossaryCore


class SectorModel():
    """
    Sector pyworld3
    General implementation of sector pyworld3
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
        self.growth_rate_df = None
        self.lt_energy_eff = None
        self.emax_enet_constraint = None
        
        self.range_energy_eff_cstrt = None
        self.energy_eff_xzero_constraint =  None 
        
    def configure_parameters(self, inputs_dict, sector_name):
        '''
        Configure with inputs_dict from the discipline
        '''
        #years range for long term energy efficiency 
        self.years_lt_energy_eff = np.arange(1950, 2120)
        self.prod_function_fitting = inputs_dict['prod_function_fitting']
        if self.prod_function_fitting:
            self.energy_eff_max_range_ref = inputs_dict['energy_eff_max_range_ref']
            self.hist_sector_invest = inputs_dict['hist_sector_investment']
        
        self.year_start = inputs_dict[GlossaryCore.YearStart]  # year start
        self.year_end = inputs_dict[GlossaryCore.YearEnd]  # year end
        self.time_step = inputs_dict[GlossaryCore.TimeStep]
        self.years_range = np.arange(self.year_start,self.year_end + 1,self.time_step)
        self.nb_years = len(self.years_range)

        self.productivity_start = inputs_dict['productivity_start']
        #self.init_gross_output = inputs_dict[GlossaryCore.InitialGrossOutput['var_name']]
        self.capital_start = inputs_dict['capital_start']
        self.productivity_gr_start = inputs_dict['productivity_gr_start']
        self.decline_rate_tfp = inputs_dict['decline_rate_tfp']
        self.depreciation_capital = inputs_dict['depreciation_capital']
        self.frac_damage_prod = inputs_dict[GlossaryCore.FractionDamageToProductivityValue]
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
        
        self.sector_name = sector_name
        
        self.init_dataframes()

    def init_dataframes(self):
        '''
        Init dataframes with years
        '''
        self.years = np.arange(self.year_start, self.year_end + 1)
        default_index = self.years
        self.capital_df = pd.DataFrame(index=default_index, columns=GlossaryCore.CapitalDf['dataframe_descriptor'].keys())
        self.production_df = pd.DataFrame(index=default_index, columns=GlossaryCore.ProductionDf['dataframe_descriptor'].keys())
        self.productivity_df = pd.DataFrame(index=default_index, columns=GlossaryCore.ProductivityDf['dataframe_descriptor'].keys())
        self.growth_rate_df = pd.DataFrame(index=default_index, columns=[GlossaryCore.Years,'net_output_growth_rate'])
        self.production_df[GlossaryCore.Years] = self.years
        self.capital_df[GlossaryCore.Years] = self.years
        self.productivity_df[GlossaryCore.Years] = self.years
        self.growth_rate_df[GlossaryCore.Years] = self.years
        self.capital_df.loc[self.year_start, GlossaryCore.Capital] = self.capital_start
    
    def set_coupling_inputs(self, inputs):
        """
        Set couplings inputs with right index, scaling... 
        """
        #If fitting takes investment from historical input not coupling
        if self.prod_function_fitting:
            self.investment_df = self.hist_sector_invest
            self.investment_df.index = self.investment_df[GlossaryCore.Years].values
        else:
            self.investment_df = inputs[GlossaryCore.InvestmentDfValue]
            self.investment_df.index = self.investment_df[GlossaryCore.Years].values
        #scale energy production
        self.energy_production = inputs[GlossaryCore.EnergyProductionValue].copy(deep=True)
        self.energy_production[GlossaryCore.TotalProductionValue] *= self.scaling_factor_energy_production
        self.energy_production.index = self.energy_production[GlossaryCore.Years].values
        self.workforce_df = inputs[GlossaryCore.WorkforceDfValue]
        self.workforce_df.index = self.workforce_df[GlossaryCore.Years].values
        self.damage_df = inputs[GlossaryCore.DamageDfValue]
        self.damage_df.index = self.damage_df[GlossaryCore.Years].values

    def compute_productivity_growthrate(self):
        '''
        A_g, Growth rate of total factor productivity without damage 
        Returns:
            :returns: A_g(0) * exp(-Δ_a * (t-1))
        '''
        t = np.arange(0, len(self.years))
        productivity_gr = self.productivity_gr_start * np.exp(-self.decline_rate_tfp * t)
        productivity_gr /= 5  
        self.productivity_df[GlossaryCore.ProductivityGrowthRate] = productivity_gr
        return productivity_gr

    def compute_productivity(self, year):
        '''
        productivity
        if damage_to_productivity= True add damage to the the productivity
        if  not: productivity evolves independently from other variables (except productivity growthrate)
        '''
        damage_to_productivity = self.damage_to_productivity
        damefrac = self.damage_df.at[year, GlossaryCore.DamageFractionOutput]
        #For year_start put initial value 
        if year == self.year_start: 
            productivity =  self.productivity_start  
        #for other years: two ways to compute:  
        elif damage_to_productivity:
            p_productivity = self.productivity_df.at[year -self.time_step, GlossaryCore.Productivity]
            p_productivity_gr = self.productivity_df.at[year - self.time_step, GlossaryCore.ProductivityGrowthRate]
            #damage = 1-damefrac
            productivity = (1 - self.frac_damage_prod * damefrac) *(p_productivity / (1 - p_productivity_gr))
        else:
            p_productivity = self.productivity_df.at[year -self.time_step, GlossaryCore.Productivity]
            p_productivity_gr = self.productivity_df.at[year - self.time_step, GlossaryCore.ProductivityGrowthRate]
            productivity = p_productivity /(1 - p_productivity_gr)
        # we divide the productivity growth rate by 5/time_step because of change in time_step (as advised in Traeger, 2013)
        self.productivity_df.loc[year, GlossaryCore.Productivity] = productivity
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
            investment = self.investment_df.loc[year - self.time_step, GlossaryCore.InvestmentsValue]
            capital = self.capital_df.at[year - self.time_step, GlossaryCore.Capital]
            capital_a = capital * (1 - self.depreciation_capital) + investment
            self.capital_df.loc[year, GlossaryCore.Capital] = capital_a
                                  
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
        capital = self.capital_df.loc[year, GlossaryCore.Capital] * 1e3
        # compute energy_efficiency
        energy_efficiency = cst + max_e / (1 + np.exp(-k * (year - xo)))
        # Then compute e_max
        e_max = capital / (capital_utilisation_ratio * energy_efficiency)

        self.capital_df.loc[year,GlossaryCore.EnergyEfficiency] = energy_efficiency
        self.capital_df.loc[year, GlossaryCore.Emax] = e_max

    def compute_usable_capital(self, year):
        """  Usable capital is the part of the capital stock that can be used in the production process. 
        To be usable the capital needs enough energy.
        K_u = K*(E/E_max) 
        E is energy in Twh and K is capital in trill dollars constant 2020
        Output: usable capital in trill dollars constant 2020
        """
        capital = self.capital_df.loc[year, GlossaryCore.Capital]

        usable_capital_unbounded = self.capital_df.loc[year, GlossaryCore.UsableCapitalUnbounded]
        upper_bound = self.max_capital_utilisation_ratio * capital

        usable_capital = upper_bound if np.real(usable_capital_unbounded) > np.real(
            upper_bound) else usable_capital_unbounded

        self.capital_df.loc[year, GlossaryCore.UsableCapital] = usable_capital

    def compute_gross_output(self, year):
        """ Compute the gdp 
        inputs: usable capital by year in trill $ , working population by year in million of people,
             productivity by year (no unit), alpha (between 0 and 1) 
        output: gdp in trillion dollars
        """
        alpha = self.output_alpha
        gamma = self.output_gamma
        productivity = self.productivity_df.loc[year, GlossaryCore.Productivity]
        working_pop = self.workforce_df.loc[year, self.sector_name]
        capital_u = self.capital_df.loc[year, GlossaryCore.UsableCapital]
        # If gamma == 1/2 use sqrt but same formula
        if gamma == 1 / 2:
            output = productivity * \
                (alpha * np.sqrt(capital_u) + (1 - alpha) * np.sqrt(working_pop))**2
        else:
            output = productivity * \
                (alpha * capital_u**gamma + (1 - alpha)* (working_pop)**gamma)**(1 / gamma)
        self.production_df.loc[year, GlossaryCore.GrossOutput] = output

        return output
    
    def compute_output_net_of_damage(self, year):
        """
        Output net of damages, trillions USD
        """
        damage_to_productivity = self.damage_to_productivity
        damefrac = self.damage_df.at[year, GlossaryCore.DamageFractionOutput]
        gross_output = self.production_df.at[year,GlossaryCore.GrossOutput]

        if damage_to_productivity:
            damage = 1 - ((1 - damefrac) / (1 - self.frac_damage_prod * damefrac))
            output_net_of_d = (1 - damage) * gross_output
        else:
            output_net_of_d = gross_output * (1 - damefrac)
        self.production_df.loc[year, GlossaryCore.OutputNetOfDamage] = output_net_of_d
        return output_net_of_d
    
    def compute_output_growth_rate(self, year):
        """ Compute output growth rate for every year for the year before: 
        output_growth_rate(t-1) = (output(t) - output(t-1))/output(t-1)
        for the last year we put the value of the previous year to avoid a 0 
        """
        if year == self.year_start: 
            pass
        else: 
            output = self.production_df.at[year - self.time_step,GlossaryCore.OutputNetOfDamage]
            output_a = self.production_df.at[year, GlossaryCore.OutputNetOfDamage]
            output = max(1e-6, output)
            output_growth = ((output_a - output) / output) / self.time_step
            self.growth_rate_df.loc[year - self.time_step, 'net_output_growth_rate'] = output_growth
        #For the last year put the vale of the year before 
        if year == self.year_end: 
            self.growth_rate_df.loc[year, 'net_output_growth_rate'] = output_growth
    
    ### CONSTRAINTS ###
    def compute_emax_enet_constraint(self):
        """ Equation for Emax constraint 
        """
        e_max = self.capital_df[GlossaryCore.Emax].values
        energy = self.energy_production[GlossaryCore.TotalProductionValue].values
        self.emax_enet_constraint = - \
            (energy - e_max * self.max_capital_utilisation_ratio) / self.ref_emax_enet_constraint

    ### For production fitting optim  only
    def compute_long_term_energy_efficiency(self):
        """ Compute energy efficiency function on a longer time scale to analyse shape
        of the function. 
        """
        #period 
        years = self.years_lt_energy_eff
        #param
        k = self.energy_eff_k
        cst = self.energy_eff_cst
        xo = self.energy_eff_xzero
        max_e = self.energy_eff_max
        # compute energy_efficiency
        energy_efficiency = cst + max_e / (1 + np.exp(-k * (years - xo)))
        self.lt_energy_eff = pd.DataFrame({GlossaryCore.Years: years, GlossaryCore.EnergyEfficiency: energy_efficiency})
        return self.lt_energy_eff
    
    def compute_energy_eff_constraints(self):
        """ 
        Compute constraints for energy efficiency fitting
        One constraint to limit the range of variation of the energy efficiency max/min < some chosen value
        One constraint to limit the value of the sigmoid midpoint (year)
        """
        #constraint for diff between min and max value
        self.range_energy_eff_cstrt = (self.energy_eff_cst + self.energy_eff_max)/self.energy_eff_cst - self.energy_eff_max_range_ref
        self.range_energy_eff_cstrt = np.array([self.range_energy_eff_cstrt])
   
        return self.range_energy_eff_cstrt

    def compute_energy_efficiency(self):
        """compute energy_efficiency"""
        years = self.capital_df[GlossaryCore.Years].values
        energy_efficiency = self.energy_eff_cst + self.energy_eff_max / (1 + np.exp(-self.energy_eff_k *
                                                                                    (years - self.energy_eff_xzero)))
        self.capital_df[GlossaryCore.EnergyEfficiency] = energy_efficiency

    def compute_unbounded_usable_capital(self):
        net_energy_production = self.energy_production[GlossaryCore.TotalProductionValue]
        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency]
        usable_capital_unbounded = self.capital_utilisation_ratio * net_energy_production * energy_efficiency * 1e-3
        self.capital_df[GlossaryCore.UsableCapitalUnbounded] = usable_capital_unbounded

    def compute_energy_usage(self):
        """Wasted energy is the overshoot of energy production not used by usable capital"""
        capital = self.capital_df[GlossaryCore.Capital]
        net_energy_production = self.energy_production[GlossaryCore.TotalProductionValue]  # PWh
        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency]
        optimal_energy_production = self.max_capital_utilisation_ratio * capital / self.capital_utilisation_ratio / energy_efficiency * 1e3
        self.productivity_df[GlossaryCore.OptimalEnergyProduction] = optimal_energy_production
        self.productivity_df[GlossaryCore.UsedEnergy] = np.minimum(net_energy_production, optimal_energy_production)
        self.productivity_df[GlossaryCore.UnusedEnergy] = np.maximum(net_energy_production - optimal_energy_production, 0.)
        # Energy_wasted = max((Enet - Eoptimal),0.)
        self.productivity_df[GlossaryCore.EnergyWasted] = (net_energy_production - optimal_energy_production) * 1e3  # TWh
        self.productivity_df.loc[self.productivity_df[GlossaryCore.EnergyWasted] < 0., GlossaryCore.EnergyWasted] = 0.

    def compute_energy_wasted_objective(self):
        """Computes normalized energy wasted constraint. Ewasted=max(Enet - Eoptimal, 0)
        Normalize by total energy since Energy wasted is around 10% of total energy => have constraint around 0.1
        which can be compared to the negative welfare objective (same order of magnitude)
        """
        # total energy is supposed to be > 0.
        energy_wasted_objective = self.productivity_df[GlossaryCore.EnergyWasted].values.sum() / \
                                  self.energy_production[GlossaryCore.TotalProductionValue].values.sum()

        self.energy_wasted_objective = np.array([energy_wasted_objective])

    #RUN
    def compute(self, inputs):
        """
        Compute all models for year range
        """
        self.init_dataframes()
        self.inputs = inputs
        self.set_coupling_inputs(inputs)
        self.compute_productivity_growthrate()
        self.compute_energy_efficiency()
        self.compute_unbounded_usable_capital()

        # iterate over years
        for year in self.years_range:
            self.compute_productivity(year)
            self.compute_emax(year)
            self.compute_usable_capital(year)
            self.compute_gross_output(year)
            self.compute_output_net_of_damage(year)
            self.compute_output_growth_rate(year)
            # capital t+1 :
            self.compute_capital(year+1)
        self.production_df = self.production_df.fillna(0.0)
        self.capital_df = self.capital_df.fillna(0.0)
        self.productivity_df = self.productivity_df.fillna(0.0)
        if self.prod_function_fitting:
            self.compute_long_term_energy_efficiency()
            self.compute_energy_eff_constraints()

        self.compute_energy_usage()
        self.compute_energy_wasted_objective()
        return self.production_df, self.capital_df, self.productivity_df, self.growth_rate_df, self.emax_enet_constraint, self.lt_energy_eff, self.range_energy_eff_cstrt
    
    ### GRADIENTS ###

    def _null_derivative(self):
        nb_years = len(self.years_range)
        return np.zeros((nb_years, nb_years))

    def _identity_derivative(self):
        nb_years = len(self.years_range)
        return np.identity(nb_years)

    def compute_doutput_dworkforce(self):
        """ Gradient for output output wrt workforce
        output = productivity * (alpha * capital_u**gamma + (1-alpha)* (working_pop)**gamma)**(1/gamma) 
        """
        years = self.years_range
        nb_years = len(years)
        alpha = self.output_alpha
        gamma = self.output_gamma
        doutput = np.identity(nb_years)
        working_pop = self.workforce_df[self.sector_name].values
        capital_u = self.capital_df[GlossaryCore.UsableCapital].values
        productivity = self.productivity_df[GlossaryCore.Productivity].values
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

    def doutput_denergy(self, dcapitalu_denergy):
        years = self.years_range
        nb_years = len(years)
        alpha = self.output_alpha
        gamma = self.output_gamma
        doutput_dcap = np.identity(nb_years)
        working_pop = self.workforce_df[self.sector_name].values
        capital_u = self.capital_df[GlossaryCore.UsableCapital].values
        productivity = self.productivity_df[GlossaryCore.Productivity].values
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
        p_productivity_gr = self.productivity_df[GlossaryCore.ProductivityGrowthRate].values
        p_productivity = self.productivity_df[GlossaryCore.Productivity].values

        # derivative matrix initialization
        d_productivity = np.zeros((nb_years, nb_years))
        if self.damage_to_productivity:

            # first line stays at zero since derivatives of initial values are
            # zero
            for i in range(1, nb_years):
                d_productivity[i, i] = (1 - self.frac_damage_prod * self.damage_df.at[years[i], GlossaryCore.DamageFractionOutput]) * \
                                    d_productivity[i - 1, i] / (1 - p_productivity_gr[i - 1]) - self.frac_damage_prod * \
                                    p_productivity[i - 1] / (1 - p_productivity_gr[i - 1])
                for j in range(1, i):
                    d_productivity[i, j] = (1 - self.frac_damage_prod * self.damage_df.at[years[i], GlossaryCore.DamageFractionOutput]) * \
                                            d_productivity[i - 1, j] / (1 - p_productivity_gr[i - 1] )

        return d_productivity
    
    def doutput_ddamage(self, dproductivity):
        years = self.years_range
        nb_years = len(years)
        alpha = self.output_alpha
        gamma = self.output_gamma
        doutput_dprod = np.identity(nb_years)
        working_pop = self.workforce_df[self.sector_name].values
        capital_u = self.capital_df[GlossaryCore.UsableCapital].values
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
        d_Ku_d_invests = self._null_derivative()
        index_zeros = self.productivity_df[GlossaryCore.UnusedEnergy].values > 0.
        for i in range(0, nb_years-1):
            for j in range(0, i + 1):
                dcapital[i + 1, j] += dcapital[i, j] * (1 - self.depreciation_capital)
                d_Ku_d_invests[i + 1, j] += index_zeros[i+1] * dcapital[i + 1, j] * self.max_capital_utilisation_ratio

        return dcapital, d_Ku_d_invests

    def d_enegy_wasted_obj_d_invest(self, d_capital_d_invest):
        index_zeros = self.productivity_df[GlossaryCore.UnusedEnergy].values > 0.
        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency].values
        d_Ew_d_invest = - np.diag(index_zeros * self.max_capital_utilisation_ratio * 1e6 / self.capital_utilisation_ratio / energy_efficiency) @ d_capital_d_invest

        sum_energy_prod = self.energy_production[GlossaryCore.TotalProductionValue].values.sum()
        d_EWO_d_EW = np.ones_like(self.years) / sum_energy_prod
        d_EWO_d_invests = d_EWO_d_EW @d_Ew_d_invest
        return d_Ew_d_invest, d_EWO_d_invests

    def doutput_dinvest(self, d_usable_capital_d_invest):
        alpha = self.output_alpha
        gamma = self.output_gamma
        working_pop = self.workforce_df[self.sector_name].values
        capital_u = self.capital_df[GlossaryCore.UsableCapital].values
        productivity = self.productivity_df[GlossaryCore.Productivity].values
        # Derivative of output wrt usable capital
        doutput_dusable_capital = np.diag(productivity * alpha * capital_u ** (gamma - 1) *
                                (alpha * capital_u ** gamma + (1 - alpha) * (working_pop) ** gamma) ** (1 / gamma - 1)
                                )
        # Then doutput = doutput_d_prod * dproductivity
        doutput_dinvests = doutput_dusable_capital @ d_usable_capital_d_invest
        return doutput_dinvests
    
    def demaxconstraint(self, dcapital):
        """ Compute derivative of e_max and emax constraint using derivative of capital. 
        For all inputs that impacts e_max through capital 
        """
        #e_max = capital*1e3/ (capital_utilisation_ratio * energy_efficiency)
        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency].values
        demax = np.identity(self.nb_years)
        demax *= 1e3 / (self.capital_utilisation_ratio * energy_efficiency)
        demax = np.dot(demax, dcapital)
        demaxconstraint_demax = demax * self. max_capital_utilisation_ratio / self.ref_emax_enet_constraint
        return demaxconstraint_demax
    
    def dnetoutput(self, doutput):
        """ Compute the derivatives of net output using derivatives of gross output
         if damage_to_productivity:
            damage = 1 - ((1 - damefrac) / (1 - self.frac_damage_prod * damefrac))
            output_net_of_d = (1 - damage) * gross_output
        else:
            output_net_of_d = gross_output * (1 - damefrac)
        """
        damefrac = self.damage_df[GlossaryCore.DamageFractionOutput].values
        if self.damage_to_productivity:
            dnet_output =(1 - damefrac) / (1 - self.frac_damage_prod * damefrac) * doutput
        else:
            dnet_output = (1 - damefrac) * doutput
        return dnet_output
    
    def dnetoutput_ddamage(self, doutput):
        """ Compute the derivatives of net output wrt damage using derivatives of gross output
         if damage_to_productivity:
            damage = 1 - ((1 - damefrac) / (1 - self.frac_damage_prod * damefrac))
            output_net_of_d = (1 - damage) * gross_output
        else:
            output_net_of_d = gross_output * (1 - damefrac)
        """
        frac = self.frac_damage_prod
        years = self.years_range
        nb_years = len(years)
        dnet_output = np.zeros((nb_years, nb_years))
        for i in range(0, nb_years):
            output = self.production_df.at[years[i], GlossaryCore.GrossOutput]
            damefrac = self.damage_df.at[years[i], GlossaryCore.DamageFractionOutput]
            for j in range(0, i + 1):
                if i == j:
                    if self.damage_to_productivity:
                        dnet_output[i, j] = (frac - 1) / ((frac * damefrac - 1)**2) * output + \
                            (1 - damefrac) / (1 - frac *damefrac) * doutput[i, j]
                    else:
                        dnet_output[i, j] = - output + (1 - damefrac) * doutput[i, j]
                else:
                    if self.damage_to_productivity:
                        dnet_output[i, j] = (1 - damefrac) / (1 - frac * damefrac) * doutput[i, j]
                    else:
                        dnet_output[i, j] = (1 - damefrac) * doutput[i, j]
        return dnet_output

    def d_Y_Ku_Ew_Constraint_d_energy(self):
        """
        Derivative of :
        - usable capital
        - Non-energy capital
        - gross output
        - lower bound constraint
        - Energy_wasted
        wrt energy
        """
        alpha = self.output_alpha
        gamma = self.output_gamma
        productivity = self.productivity_df[GlossaryCore.Productivity].values
        working_pop = self.workforce_df[self.sector_name].values
        usable_capital = self.capital_df[GlossaryCore.UsableCapital].values

        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency].values
        d_UKu_d_E = np.diag(self.capital_utilisation_ratio * energy_efficiency)
        d_Ku_d_E = np.diag(self.capital_utilisation_ratio * energy_efficiency)

        index_zeros = self.productivity_df[GlossaryCore.UnusedEnergy].values > 0.

        d_Ku_d_E[index_zeros, index_zeros] = 0.

        damefrac = self.damage_df[GlossaryCore.DamageFractionOutput].values
        dQ_dY = 1 - damefrac if not self.damage_to_productivity else (1 - damefrac) / (1 - self.frac_damage_prod * damefrac)
        d_K_d_E = self._null_derivative()
        dY_dE = np.diag(
            productivity * alpha * usable_capital ** (gamma - 1) * np.diag(d_Ku_d_E) *
            (alpha * usable_capital ** gamma + (1 - alpha) * working_pop ** gamma) ** (1. / gamma - 1.)
        )
        dY_dE[0, 0] = 0.

        for i in range(1, self.nb_years + 1):
            for j in range(1, i):
                dY_dE[i - 1, j] = productivity[i-1] * alpha * usable_capital[i-1] ** (gamma - 1) *\
                    d_Ku_d_E[i - 1, j] * (alpha * usable_capital[i-1] ** gamma + (1 - alpha) * working_pop[i-1] ** gamma) ** (1. / gamma - 1.)
                if i < self.nb_years:
                    d_K_d_E[i, j] = (1 - self.depreciation_capital) * d_K_d_E[i - 1, j]
                    d_Ku_d_E[i, j] = index_zeros[i] * self.max_capital_utilisation_ratio * d_K_d_E[i, j]


        # Energy_wasted Ew = E - KNE * k where k = max_capital_utilisation_ratio/capital_utilisation_ratio/energy_efficiency*1.e3
        # energy_efficiency is function of the years. Eoptimal in TWh
        k = np.diag(self.max_capital_utilisation_ratio / self.capital_utilisation_ratio / energy_efficiency * 1.e3)
        d_energy_wasted_d_energy = self._identity_derivative() * 1.e3 - np.matmul(k, d_K_d_E) # Enet converted from PWh to TWh
        # Since Ewasted = max(Enet - Eoptimal, 0.), gradient should be 0 when Enet - Eoptimal <=0, ie when Ewasted =0
        # => put to 0 the lines of the gradient matrix corresponding to the years where Ewasted=0
        matrix_of_years_E_is_wasted = (self.productivity_df[GlossaryCore.EnergyWasted].values > 0.).astype(int)
        d_energy_wasted_d_energy = np.transpose(np.multiply(matrix_of_years_E_is_wasted, np.transpose(d_energy_wasted_d_energy)))
        d_sum_energy_wasted_d_energy_total = np.ones(self.nb_years) @ d_energy_wasted_d_energy
        d_sum_energy_total_d_energy_total = np.ones(self.nb_years) @ self._identity_derivative()

        sum_ewasted = self.productivity_df[GlossaryCore.EnergyWasted].values.sum()
        sum_etotal = self.energy_production[GlossaryCore.TotalProductionValue].values.sum()
        # sumetotal is supposed > 0 otherwise no energy in the system => cannot work
        grad_energy_wasted_obj = (sum_etotal * d_sum_energy_wasted_d_energy_total - sum_ewasted * d_sum_energy_total_d_energy_total) / \
                                 sum_etotal ** 2 * 1e3

        return dY_dE, d_UKu_d_E,d_Ku_d_E, grad_energy_wasted_obj
