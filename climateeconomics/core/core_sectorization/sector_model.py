'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/29-2023/11/06 Copyright 2023 Capgemini

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
        self.usable_capital_upper_bound_constraint = None
        self.energy_production = None
        self.productivity_df = None
        self.capital_df = None
        self.production_df = None
        self.damage_df = None
        self.workforce_df = None
        self.growth_rate_df = None
        self.lt_energy_eff = None
        self.emax_enet_constraint = None
        self.gdp_percentage_per_section_df = None
        self.section_list = []
        self.section_gdp_df = None
        self.energy_consumption_percentage_per_section_df = None
        self.section_energy_consumption_df = None
        self.range_energy_eff_cstrt = None
        self.energy_eff_xzero_constraint = None
        self.usable_capital_ref = None
        self.usable_capital_objective_ref = None

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
        self.years_range = np.arange(self.year_start,self.year_end + 1)
        self.nb_years = len(self.years_range)
        self.sector_name = sector_name
        self.section_list = GlossaryCore.SectionDictSectors[self.sector_name]
        self.gdp_percentage_per_section_df = inputs_dict[f"{self.sector_name}.{GlossaryCore.SectionGdpPercentageDfValue}"]
        self.energy_consumption_percentage_per_section_df = inputs_dict[f"{self.sector_name}.{GlossaryCore.SectionEnergyConsumptionPercentageDfValue}"]
        def correct_years(df):

            input_dict = {GlossaryCore.Years: self.years_range}
            input_dict.update({section: df[section].values[0] for section in self.section_list})
            return pd.DataFrame(input_dict)
        self.gdp_percentage_per_section_df = correct_years(self.gdp_percentage_per_section_df)
        self.energy_consumption_percentage_per_section_df = correct_years(self.energy_consumption_percentage_per_section_df)
        self.productivity_start = inputs_dict['productivity_start']
        self.capital_start = inputs_dict['capital_start']
        self.productivity_gr_start = inputs_dict['productivity_gr_start']
        self.decline_rate_tfp = inputs_dict['decline_rate_tfp']
        self.depreciation_capital = inputs_dict['depreciation_capital']
        self.frac_damage_prod = inputs_dict[GlossaryCore.FractionDamageToProductivityValue]
        self.damage_to_productivity = inputs_dict[GlossaryCore.DamageToProductivity]
        self.output_alpha = inputs_dict['output_alpha']
        self.output_gamma = inputs_dict['output_gamma']
        self.energy_eff_k = inputs_dict['energy_eff_k']
        self.energy_eff_cst = inputs_dict['energy_eff_cst']
        self.energy_eff_xzero = inputs_dict['energy_eff_xzero']
        self.energy_eff_max = inputs_dict['energy_eff_max']
        self.capital_utilisation_ratio = inputs_dict['capital_utilisation_ratio']
        self.max_capital_utilisation_ratio = inputs_dict['max_capital_utilisation_ratio']
        self.ref_emax_enet_constraint = inputs_dict['ref_emax_enet_constraint']
        self.compute_climate_impact_on_gdp = inputs_dict['assumptions_dict']['compute_climate_impact_on_gdp']
        if not self.compute_climate_impact_on_gdp:
            self.damage_to_productivity = False
        
        self.sector_name = sector_name
        
        self.init_dataframes()
        self.usable_capital_ref = inputs_dict["usable_capital_ref"]

    def init_dataframes(self):
        '''
        Init dataframes with years
        '''
        self.years = np.arange(self.year_start, self.year_end + 1)
        default_index = self.years
        self.capital_df = pd.DataFrame(index=default_index, columns=GlossaryCore.CapitalDf['dataframe_descriptor'].keys(), dtype=float)
        self.production_df = pd.DataFrame(index=default_index, columns=GlossaryCore.ProductionDf['dataframe_descriptor'].keys(), dtype=float)
        self.section_gdp_df = pd.DataFrame(index=default_index, columns=GlossaryCore.SectionGdpDf['dataframe_descriptor'].keys(), dtype=float)
        self.damage_df = pd.DataFrame(index=default_index, columns=GlossaryCore.DamageDetailedDf['dataframe_descriptor'].keys(), dtype=float)
        self.productivity_df = pd.DataFrame(index=default_index, columns=GlossaryCore.ProductivityDf['dataframe_descriptor'].keys(), dtype=float)
        self.growth_rate_df = pd.DataFrame(index=default_index, columns=[GlossaryCore.Years, 'net_output_growth_rate'])
        self.production_df[GlossaryCore.Years] = self.years
        self.section_gdp_df[GlossaryCore.Years] = self.years
        self.damage_df[GlossaryCore.Years] = self.years
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
            self.investment_df = inputs[f"{self.sector_name}.{GlossaryCore.InvestmentDfValue}"]
            self.investment_df.index = self.investment_df[GlossaryCore.Years].values
        #scale energy production
        self.energy_production = inputs[GlossaryCore.EnergyProductionValue]
        self.workforce_df = inputs[GlossaryCore.WorkforceDfValue]
        self.workforce_df.index = self.workforce_df[GlossaryCore.Years].values
        self.damage_fraction_output_df = inputs[GlossaryCore.DamageFractionDfValue]
        self.damage_fraction_output_df.index = self.damage_fraction_output_df[GlossaryCore.Years].values

    def compute_productivity_growthrate(self):
        """
        A_g, Growth rate of total factor productivity.
        Returns:
            :returns: A_g(0) * exp(-Δ_a * (t-1))
        """
        prod_growth_rate = self.productivity_gr_start * np.exp(
            - self.decline_rate_tfp * (self.years_range - self.year_start))
        self.productivity_df[GlossaryCore.ProductivityGrowthRate] = prod_growth_rate

    def compute_productivity(self):
        """
        productivity
        if damage_to_productivity= True add damage to the the productivity
        if  not: productivity evolves independently from other variables (except productivity growthrate)
        """
        prod_wo_d = self.productivity_start
        productivity_wo_damage_list = [self.productivity_start]

        damage_fraction_output = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput].values
        productivity_growth = self.productivity_df[GlossaryCore.ProductivityGrowthRate].values

        for prod_growth, damage_frac_year in zip(productivity_growth[:-1], damage_fraction_output[1:]):
            prod_wo_d = prod_wo_d / (1 - prod_growth / 5)
            productivity_wo_damage_list.append(prod_wo_d)

        productivity_w_damage_list = np.array(productivity_wo_damage_list) * (1 - damage_fraction_output)

        self.productivity_df[GlossaryCore.ProductivityWithDamage] = productivity_w_damage_list
        self.productivity_df[GlossaryCore.ProductivityWithoutDamage] = productivity_wo_damage_list
        if self.damage_to_productivity:
            self.productivity_df[GlossaryCore.Productivity] = productivity_w_damage_list
        else:
            self.productivity_df[GlossaryCore.Productivity] = productivity_wo_damage_list

    def compute_capital(self):
        """
        K(t), Capital for time period, trillions $USD
        Args:
            :param capital: capital
            :param depreciation: depreciation rate
            :param investment: investment
            K(t) = K(t-1)*(1-depre_rate) + investment(t-1)
        """
        capital = self.capital_start
        capital_list = [capital]

        investments = self.investment_df[GlossaryCore.InvestmentsValue].values
        for invest in investments[:-1]:
            capital = (1 - self.depreciation_capital) * capital + invest
            capital_list.append(capital)

        capital = np.array(capital_list)

        self.capital_df[GlossaryCore.Capital] = capital

    def compute_usable_capital(self):
        """
        Usable capital = capital utilisation ratio * energy efficiency * energy production
        """
        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency].values
        energy_production = self.energy_production[GlossaryCore.TotalProductionValue].values
        usable_capital = self.capital_utilisation_ratio * energy_efficiency * energy_production
        self.capital_df[GlossaryCore.UsableCapital] = usable_capital

    def compute_gross_output(self):
        """ Compute the gdp 
        inputs: usable capital by year in trill $ , working population by year in million of people,
             productivity by year (no unit), alpha (between 0 and 1) 
        output: gdp in trillion dollars
        """
        alpha = self.output_alpha
        gamma = self.output_gamma
        productivity = self.productivity_df[GlossaryCore.Productivity].values
        working_pop = self.workforce_df[self.sector_name].values
        capital_u = self.capital_df[GlossaryCore.UsableCapital].values
        output = productivity * (alpha * capital_u**gamma + (1 - alpha)* (working_pop)**gamma) **(1 / gamma)
        self.production_df[GlossaryCore.GrossOutput] = output

    def compute_output_net_of_damage(self):
        """
        Output net of damages, trillions USD
        """
        damage_to_productivity = self.damage_to_productivity
        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput].values
        gross_output = self.production_df[GlossaryCore.GrossOutput].values

        if not self.compute_climate_impact_on_gdp:
            output_net_of_d = gross_output
        else:
            if damage_to_productivity:
                damage = 1 - ((1 - damefrac) / (1 - self.frac_damage_prod * damefrac))
                output_net_of_d = (1 - damage) * gross_output
            else:
                output_net_of_d = gross_output * (1 - damefrac)
        self.production_df[GlossaryCore.OutputNetOfDamage] = output_net_of_d

    def compute_output_net_of_damage_per_section(self):
        """
        Splitting output net of damages between sections of the sector
        """
        section_gdp_df = {GlossaryCore.Years: self.years}
        for section in self.section_list:
            section_gdp_df[section] = self.production_df[GlossaryCore.OutputNetOfDamage].values /100. * self.gdp_percentage_per_section_df[section]

        self.section_gdp_df = pd.DataFrame(section_gdp_df)

    def compute_output_growth_rate(self):
        """ Compute output growth rate for every year for the year before: 
        output_growth_rate(t-1) = (output(t) - output(t-1))/output(t-1)
        for the last year we put the value of the previous year to avoid a 0 
        """
        gross_output = self.production_df[GlossaryCore.GrossOutput]
        self.production_df[GlossaryCore.OutputGrowth] = (gross_output.diff() / gross_output.shift(1)).fillna(0.) * 100

    # For production fitting optim  only
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

    def compute_damage_from_productivity_loss(self):
        """
        Compute damages due to loss of productivity.

        As GDP ~= productivity x (Usable capital + Labor), and that we can compute productivity with or without damages,
        we compute the damages on GDP from loss of productivity as
        (productivity wo damage - productivity w damage) x (Usable capital + Labor).
        """
        productivity_w_damage = self.productivity_df[GlossaryCore.ProductivityWithDamage].values
        productivity_wo_damage = self.productivity_df[GlossaryCore.ProductivityWithoutDamage].values
        gross_output = self.production_df[GlossaryCore.GrossOutput].values
        applied_productivity = self.productivity_df[GlossaryCore.Productivity].values

        estimated_damage_from_productivity_loss = (productivity_wo_damage - productivity_w_damage) / applied_productivity * gross_output
        if self.damage_to_productivity:
            damage_from_productivity_loss = estimated_damage_from_productivity_loss
        else:
            damage_from_productivity_loss = np.zeros_like(estimated_damage_from_productivity_loss)

        self.damage_df[GlossaryCore.DamagesFromProductivityLoss] = damage_from_productivity_loss
        self.damage_df[GlossaryCore.EstimatedDamagesFromProductivityLoss] = estimated_damage_from_productivity_loss

    def compute_damage_from_climate(self):
        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput]
        gross_output = self.production_df[GlossaryCore.GrossOutput].values
        net_output = self.production_df[GlossaryCore.OutputNetOfDamage].values

        damage_from_climate = np.zeros_like(gross_output)
        if self.compute_climate_impact_on_gdp:
            damage_from_climate = gross_output - net_output
            estimated_damage_from_climate = damage_from_climate
        else:
            if self.damage_to_productivity:
                estimated_damage_from_climate = gross_output * damefrac * (1 - self.frac_damage_prod) / (
                        1 - self.frac_damage_prod * damefrac)
            else:
                estimated_damage_from_climate = gross_output * damefrac

        self.damage_df[GlossaryCore.DamagesFromClimate] = damage_from_climate
        self.damage_df[GlossaryCore.EstimatedDamagesFromClimate] = estimated_damage_from_climate

    def compute_total_damages(self):
        """Damages are the sum of damages from climate + damges from loss of productivity"""

        self.damage_df[GlossaryCore.EstimatedDamages] = self.damage_df[GlossaryCore.EstimatedDamagesFromClimate] + self.damage_df[GlossaryCore.EstimatedDamagesFromProductivityLoss]
        self.damage_df[GlossaryCore.Damages] = self.damage_df[GlossaryCore.DamagesFromClimate] + self.damage_df[GlossaryCore.DamagesFromProductivityLoss]

    def compute_energy_consumption_per_section(self):
        """
        Computing the energy consumption for each section of the sector

        section_energy_consumption (PWh) = sector_energy_production (Pwh) x section_energy_consumption_percentage (%)
        """
        section_energy_consumption = {
            GlossaryCore.Years: self.years
        }
        sector_energy_production = self.energy_production[GlossaryCore.TotalProductionValue].values
        for section in self.section_list:
            section_energy_consumption[section] = sector_energy_production * self.energy_consumption_percentage_per_section_df[section].values / 100.
        self.section_energy_consumption_df = pd.DataFrame(section_energy_consumption)

    def compute_usable_capital_upper_bound_constraint(self):
        """
        Upper bound usable capital constraint = max capital utilisation ratio * non energy capital - usable capital
        """
        ne_capital = self.capital_df[GlossaryCore.Capital].values
        usable_capital = self.capital_df[GlossaryCore.UsableCapital].values
        self.usable_capital_upper_bound_constraint = - (usable_capital - self.max_capital_utilisation_ratio * ne_capital) / self.usable_capital_ref

    # RUN
    def compute(self, inputs):
        """
        Compute all models for year range
        """
        self.init_dataframes()
        self.set_coupling_inputs(inputs)
        self.compute_productivity_growthrate()
        self.compute_productivity()
        self.compute_energy_efficiency()
        self.compute_usable_capital()

        self.compute_gross_output()
        self.compute_output_net_of_damage()
        self.compute_output_growth_rate()
        self.compute_capital()

        if self.prod_function_fitting:
            self.compute_long_term_energy_efficiency()
            self.compute_energy_eff_constraints()

        self.compute_output_net_of_damage_per_section()
        self.compute_energy_consumption_per_section()

        self.compute_usable_capital_upper_bound_constraint()
        self.compute_damage_from_productivity_loss()
        self.compute_damage_from_climate()
        self.compute_total_damages()
        
        self.output_types_to_float()

    
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
        doutput = doutput @ np.diag(f_prime)
        return doutput

    def d_energy_production(self):
        """
        Derivative of :
        - usable capital
        - gross output
        wrt energy
        """
        alpha = self.output_alpha
        gamma = self.output_gamma
        productivity = self.productivity_df[GlossaryCore.Productivity].values
        working_pop = self.workforce_df[self.sector_name].values
        usable_capital = self.capital_df[GlossaryCore.UsableCapital].values

        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency].values
        d_usable_capital_d_energy = np.diag(self.capital_utilisation_ratio * energy_efficiency)

        d_gross_output_d_energy = d_usable_capital_d_energy @ np.diag(
            productivity * alpha * usable_capital ** (gamma - 1) * (alpha * usable_capital ** gamma + (1 - alpha) * working_pop ** gamma) ** (1. / gamma - 1.)
        )

        d_net_output_d_energy = self._d_net_output_d_user_input(d_gross_output_d_energy)

        d_damages_d_energy, d_estimated_damages_d_energy = self.d_damages_d_user_input(d_gross_output_d_energy, d_net_output_d_energy)

        d_k_d_energy = self._null_derivative()
        d_ku_ub_contraint = self.d_ku_upper_bound_constraint_d_user_input(d_usable_capital_d_energy, d_k_d_energy)
        d_damages_from_climate = self.__d_damages_from_climate_d_user_input(d_gross_output_d_energy, d_net_output_d_energy)
        a = {
            GlossaryCore.Capital: d_k_d_energy,
            GlossaryCore.Damages: d_damages_d_energy,
            GlossaryCore.EstimatedDamages: d_estimated_damages_d_energy,
            GlossaryCore.ConstraintUpperBoundUsableCapital: d_ku_ub_contraint,
            GlossaryCore.UsableCapital: d_usable_capital_d_energy,
            GlossaryCore.GrossOutput: d_gross_output_d_energy,
            GlossaryCore.OutputNetOfDamage: d_net_output_d_energy,
            GlossaryCore.DamagesFromClimate: d_damages_from_climate,
        }
        return d_gross_output_d_energy, d_net_output_d_energy, d_estimated_damages_d_energy, d_damages_d_energy, d_ku_ub_contraint, d_usable_capital_d_energy

    def d_working_pop(self):
        """
        Derivative of :
        - usable capital
        - gross output
        - lower bound constraint
        wrt working age population
        """
        years = self.years_range
        nb_years = len(years)
        alpha = self.output_alpha
        gamma = self.output_gamma
        doutput = np.identity(nb_years)
        working_pop = self.workforce_df[self.sector_name].values
        capital_u = self.capital_df[GlossaryCore.UsableCapital].values
        productivity = self.productivity_df[GlossaryCore.Productivity].values
        g = alpha * capital_u ** gamma + (1 - alpha) * (working_pop) ** gamma
        g_prime = (1 - alpha) * gamma * working_pop ** (gamma - 1)
        f_prime = productivity * (1 / gamma) * g * g_prime
        d_gross_output_d_wap = doutput @ np.diag(f_prime)
        d_usable_capital_d_wap = self._null_derivative()
        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput].values
        dQ_dY = 1 - damefrac if not self.damage_to_productivity else (1 - damefrac) / (1 - self.frac_damage_prod * damefrac)
        if not self.compute_climate_impact_on_gdp:
            dQ_dY = np.ones_like(self.years_range)
        d_net_output_d_wap = np.diag(dQ_dY) @ d_gross_output_d_wap

        d_damages_d_wap, d_estimated_damages_d_wap = self.d_damages_d_user_input(d_gross_output_d_wap, d_net_output_d_wap)

        d_k_d_wap = self._null_derivative()
        d_ku_constraint_d_wap = self.d_ku_upper_bound_constraint_d_user_input(self._null_derivative(), d_k_d_wap)
        a = {
            GlossaryCore.Capital: d_k_d_wap,
            GlossaryCore.Damages: d_damages_d_wap,
            GlossaryCore.EstimatedDamages: d_estimated_damages_d_wap,
            GlossaryCore.ConstraintUpperBoundUsableCapital: d_ku_constraint_d_wap,
            GlossaryCore.UsableCapital: d_usable_capital_d_wap,
            GlossaryCore.GrossOutput: d_gross_output_d_wap,
            GlossaryCore.OutputNetOfDamage: d_net_output_d_wap,
        }
        return d_gross_output_d_wap, d_net_output_d_wap, d_damages_d_wap, d_estimated_damages_d_wap, d_ku_constraint_d_wap

    def d_damage_frac_output(self):
        """derivative of net output wrt damage frac output"""
        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput].values
        gross_output = self.production_df[GlossaryCore.GrossOutput].values
        productivity = self.productivity_df[GlossaryCore.Productivity].values

        d_gross_output_d_dfo =  np.diag(gross_output / productivity) @ self.d_productivity_d_damage_frac_output()

        if self.compute_climate_impact_on_gdp and self.damage_to_productivity:
            factor = (1 - damefrac) / (1 - self.frac_damage_prod * damefrac)
            d_factor_d_dfo = np.diag((self.frac_damage_prod - 1) / (1 - self.frac_damage_prod * damefrac) ** 2)
        elif self.compute_climate_impact_on_gdp and not self.damage_to_productivity:
            factor = 1 - damefrac
            d_factor_d_dfo = -np.eye(self.nb_years)
        elif not self.compute_climate_impact_on_gdp:
            factor = np.ones_like(damefrac)
            d_factor_d_dfo = self._null_derivative()
        else:
            raise Exception("Problem")
        d_net_output_d_dfo = np.diag(gross_output) @ d_factor_d_dfo + np.diag(factor) @ d_gross_output_d_dfo

        d_damages_from_productivity_loss_d_dfo, d_estimated_damages_from_productivity_loss_d_dfo = self.d_damages_from_productivity_loss_d_damage_fraction_output(d_gross_output_d_dfo)
        d_estimated_damages_from_climate_d_dfo = self.d_estimated_damages_from_climate_d_damage_frac_output(d_gross_output_d_dfo, d_net_output_d_dfo)
        d_damages_from_climate_d_dfo = self.__d_damages_from_climate_d_user_input(d_gross_output_d_dfo, d_net_output_d_dfo)
        d_estimated_damages_d_dfo = d_estimated_damages_from_climate_d_dfo + d_estimated_damages_from_productivity_loss_d_dfo
        d_damages_d_dfo = d_damages_from_climate_d_dfo + d_damages_from_productivity_loss_d_dfo

        d_k_d_dfo = self._null_derivative()
        dku_ub_constraint_d_dfo = self.d_ku_upper_bound_constraint_d_user_input(self._null_derivative(), d_k_d_dfo)
        a = {
            GlossaryCore.Capital: d_k_d_dfo,
            GlossaryCore.Damages: d_damages_d_dfo,
            GlossaryCore.EstimatedDamages: d_estimated_damages_d_dfo,
            GlossaryCore.ConstraintUpperBoundUsableCapital: dku_ub_constraint_d_dfo,
            GlossaryCore.UsableCapital: self._null_derivative(),
            GlossaryCore.GrossOutput: d_gross_output_d_dfo,
            GlossaryCore.OutputNetOfDamage: d_net_output_d_dfo,
        }
        return d_gross_output_d_dfo, d_net_output_d_dfo, d_estimated_damages_d_dfo, d_damages_d_dfo, dku_ub_constraint_d_dfo

    def d_invests(self):
        """ Compute derivative of capital wrt investments.
        """
        nb_years = self.nb_years
        d_capital_d_invests = self._null_derivative()
        for i in range(nb_years):
            for j in range(nb_years):
                d_capital_d_invests[i, j] = d_capital_d_invests[i-1, j] * (1 - self.depreciation_capital) + 1 * (i - 1 == j)

        d_ku_constraint_d_invests = self.max_capital_utilisation_ratio * d_capital_d_invests / self.usable_capital_ref
        return d_capital_d_invests, d_ku_constraint_d_invests

    def _d_net_output_d_user_input(self, d_gross_output_d_user_input):
        """derivative of net output wrt X, user should provide the derivative of gross output wrt X"""
        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput].values

        if self.compute_climate_impact_on_gdp and self.damage_to_productivity:
            dQ_dY = np.diag((1 - damefrac) / (1 - self.frac_damage_prod * damefrac))
        elif self.compute_climate_impact_on_gdp and not self.damage_to_productivity:
            dQ_dY = np.diag(1 - damefrac)
        elif not self.compute_climate_impact_on_gdp:
            dQ_dY = np.eye(self.nb_years)
        else:
            raise Exception("Problem")
        dQ_d_user_input = dQ_dY @ d_gross_output_d_user_input
        return dQ_d_user_input
    
    def d_productivity_w_damage_d_damage_frac_output(self):
        """derivative of productivity with damage wrt damage frac output"""
        productivity_wo_damage = self.productivity_df[GlossaryCore.ProductivityWithoutDamage].values
        return np.diag(-productivity_wo_damage)

    def d_productivity_d_damage_frac_output(self):
        """gradient for productivity for damage_df"""
        d_productivity_d_damage_frac_output = self._null_derivative()

        if self.damage_to_productivity:
            d_productivity_d_damage_frac_output = self.d_productivity_w_damage_d_damage_frac_output()
        return d_productivity_d_damage_frac_output

    def __d_damages_from_climate_d_user_input(self, d_gross_output_d_user_input, d_net_output_d_user_input):
        """
        damages_from_climate = gross output - net output
        """
        derivative = d_gross_output_d_user_input - d_net_output_d_user_input
        return derivative

    def __d_estimated_damages_from_climate_d_user_input(self, d_gross_output_d_user_input, d_net_output_d_user_input):
        """
        damages_from_climate = gross output - net output
        """
        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput]

        if self.compute_climate_impact_on_gdp:
            derivative = d_gross_output_d_user_input - d_net_output_d_user_input
        else:
            if self.damage_to_productivity:
                derivative = np.diag(damefrac * (1 - self.frac_damage_prod) /
                                     (1 - self.frac_damage_prod * damefrac)) @ d_gross_output_d_user_input
            else:
                derivative = np.diag(damefrac) @ d_gross_output_d_user_input

        return derivative

    def d_estimated_damages_from_climate_d_damage_frac_output(self, d_gross_output_d_user_input, d_net_output_d_user_input):
        """
        damages_from_climate = gross output - net output
        """
        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput]
        gross_output = self.production_df[GlossaryCore.GrossOutput].values

        if self.compute_climate_impact_on_gdp:
            derivative = d_gross_output_d_user_input - d_net_output_d_user_input
        else:
            if self.damage_to_productivity:
                derivative = d_gross_output_d_user_input @ np.diag(damefrac * (1 - self.frac_damage_prod) /
                                     (1 - self.frac_damage_prod * damefrac)) + np.diag(gross_output) * \
                             np.diag((1 - self.frac_damage_prod) / (1 - self.frac_damage_prod * damefrac) ** 2)

            else:
                derivative = d_gross_output_d_user_input @ np.diag(damefrac) + np.diag(gross_output)

        return derivative

    def d_damages_from_productivity_loss_d_damage_fraction_output(self, d_gross_output_d_damage_fraction_output):
        gross_output = self.production_df[GlossaryCore.GrossOutput].values
        productivity_wo_damage = self.productivity_df[GlossaryCore.ProductivityWithoutDamage].values
        productivity_w_damage = self.productivity_df[GlossaryCore.ProductivityWithDamage].values

        d_productivity_w_damage_d_damage_frac_output = self.d_productivity_w_damage_d_damage_frac_output()
        nb_years = len(self.years_range)
        d_damages_from_productivity_loss_d_damage_fraction_output = self._null_derivative()
        d_estimated_damages_from_productivity_loss_d_damage_fraction_output = self._null_derivative()
        if self.damage_to_productivity:
            for i in range(nb_years):
                d_estimated_damages_from_productivity_loss_d_damage_fraction_output[i] = d_gross_output_d_damage_fraction_output[i] * (productivity_wo_damage[i]/productivity_w_damage[i] - 1) +\
                    - gross_output[i] * productivity_wo_damage[i] / productivity_w_damage[i] ** 2 * d_productivity_w_damage_d_damage_frac_output[i]
        else:
            d_estimated_damages_from_productivity_loss_d_damage_fraction_output = np.diag((productivity_wo_damage - productivity_w_damage)/productivity_wo_damage) @ d_gross_output_d_damage_fraction_output - np.diag(gross_output / productivity_wo_damage) @ d_productivity_w_damage_d_damage_frac_output
        if self.compute_climate_impact_on_gdp and self.damage_to_productivity:
            d_damages_from_productivity_loss_d_damage_fraction_output = d_estimated_damages_from_productivity_loss_d_damage_fraction_output

        return d_damages_from_productivity_loss_d_damage_fraction_output, d_estimated_damages_from_productivity_loss_d_damage_fraction_output

    def __d_damages_from_productivity_loss_d_user_input(self, d_gross_output_d_user_input):
        productivity_wo_damage = self.productivity_df[GlossaryCore.ProductivityWithoutDamage].values
        productivity_w_damage = self.productivity_df[GlossaryCore.ProductivityWithDamage].values

        d_damages_from_productivity_loss_d_user_input = self._null_derivative()
        applied_productivity = self.productivity_df[GlossaryCore.Productivity].values
        d_estimated_damages_from_prod_loss_d_user_input = np.diag((productivity_wo_damage - productivity_w_damage) / (
                applied_productivity)) @  d_gross_output_d_user_input

        if self.compute_climate_impact_on_gdp and self.damage_to_productivity:
            d_damages_from_productivity_loss_d_user_input = d_estimated_damages_from_prod_loss_d_user_input

        return d_damages_from_productivity_loss_d_user_input, d_estimated_damages_from_prod_loss_d_user_input

    def d_damages_d_user_input(self, d_gross_output_d_user_input, d_net_output_d_user_input):
        d_damages_from_climate = self.__d_damages_from_climate_d_user_input(d_gross_output_d_user_input, d_net_output_d_user_input)
        d_estimated_damages_from_climate = self.__d_estimated_damages_from_climate_d_user_input(d_gross_output_d_user_input, d_net_output_d_user_input)
        d_damages_from_prod_loss, d_estimated_damages_from_prod_loss = self.__d_damages_from_productivity_loss_d_user_input(d_gross_output_d_user_input)
        d_damages_d_user_input = d_damages_from_prod_loss + d_damages_from_climate
        d_estimated_damages_d_user_input = d_estimated_damages_from_climate + d_estimated_damages_from_prod_loss

        return d_damages_d_user_input, d_estimated_damages_d_user_input

    def output_types_to_float(self):
        """make sure these dataframes columns have type float instead of object to avoid errors during
        seting of partial derivatives"""
        dataframes = [
            self.production_df,
            self.section_gdp_df,
            self.damage_df,
            self.capital_df,
            self.productivity_df
        ]

        for df in dataframes:
            df.fillna(0.0, inplace=True)

    def d_ku_upper_bound_constraint_d_user_input(self, d_ku_d_user_input, d_kne_d_user_input):
        return - (d_ku_d_user_input - self.max_capital_utilisation_ratio * d_kne_d_user_input) / self.usable_capital_ref