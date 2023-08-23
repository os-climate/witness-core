"""
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
"""
import numpy as np
import pandas as pd
from copy import deepcopy
from sostrades_core.tools.base_functions.exp_min import compute_func_with_exp_min
from sostrades_core.tools.cst_manager.constraint_manager import compute_delta_constraint
from climateeconomics.glossarycore import GlossaryCore


class MacroEconomics():
    """
    Economic pyworld3 that compute the evolution of capital, consumption, output...
    """
    PC_CONSUMPTION_CONSTRAINT = 'pc_consumption_constraint'

    def __init__(self, param):
        """
        Constructor
        """
        self.param = param
        self.inputs = None
        self.economics_df = None

        self.damefrac = None
        self.scaling_factor_energy_production = None
        self.energy_production = None
        self.co2_emissions_Gt = None
        self.co2_taxes = None
        self.share_energy_investment = None
        self.share_non_energy_investment = None
        self.energy_capital = None
        self.population_df = None
        self.working_age_population_df = None
        self.compute_gdp = None
        self.gross_output_in = None
        self.year_start = None
        self.year_end = None
        self.time_step = None
        self.productivity_start = None
        self.init_gross_output = None
        self.capital_start_ne = None
        self.productivity_gr_start = None
        self.decline_rate_tfp = None
        self.depreciation_capital = None
        self.init_rate_time_pref = None
        self.conso_elasticity = None
        self.lo_capital = None
        self.lo_conso = None
        self.lo_per_capita_conso = None
        self.hi_per_capita_conso = None
        self.ref_pc_consumption_constraint = None
        self.nb_per = None
        self.years_range = None
        self.nb_years = None
        self.frac_damage_prod = None
        self.damage_to_productivity = None
        self.init_output_growth = None
        self.output_alpha = None
        self.output_gamma = None
        self.energy_eff_k = None
        self.energy_eff_cst = None
        self.energy_eff_xzero = None
        self.co2_invest_limit = None
        self.employment_a_param = None
        self.employment_power_param = None
        self.employment_rate_base_value = None
        self.ref_emax_enet_constraint = None
        self.usable_capital_ref = None
        self.invest_co2_tax_in_renawables = None
        self.energy_investment_wo_renewable = None
        self.energy_investment = None
        self.workforce_df = None
        self.capital_df = None
        self.energy_eff_max = None
        self.capital_utilisation_ratio = None
        self.max_capital_utilisation_ratio = None
        self.co2_tax_efficiency = None
        self.scaling_factor_energy_investment = None
        self.alpha = None
        self.delta_capital_cons_limit = None

        self.set_data()
        self.create_dataframe()

    def set_data(self):
        self.year_start = self.param['year_start']
        self.year_end = self.param['year_end']
        self.time_step = self.param['time_step']

        self.productivity_start = self.param['productivity_start']
        self.init_gross_output = self.param['init_gross_output']
        self.capital_start_ne = self.param['capital_start_non_energy']
        self.population_df = self.param['population_df']
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
        self.nb_years = len(self.years_range)
        self.frac_damage_prod = self.param['frac_damage_prod']
        self.damage_to_productivity = self.param['damage_to_productivity']
        self.init_output_growth = self.param['init_output_growth']
        self.output_alpha = self.param['output_alpha']
        self.output_gamma = self.param['output_gamma']
        self.energy_eff_k = self.param['energy_eff_k']
        self.energy_eff_cst = self.param['energy_eff_cst']
        self.energy_eff_xzero = self.param['energy_eff_xzero']
        self.energy_eff_max = self.param['energy_eff_max']
        self.capital_utilisation_ratio = self.param['capital_utilisation_ratio']
        self.max_capital_utilisation_ratio = self.param['max_capital_utilisation_ratio']
        self.co2_emissions_Gt = self.param['co2_emissions_Gt']
        self.co2_taxes = self.param['CO2_taxes']
        self.co2_tax_efficiency = self.param['CO2_tax_efficiency']
        self.scaling_factor_energy_investment = self.param['scaling_factor_energy_investment']
        self.alpha = self.param['alpha']
        self.delta_capital_cons_limit = self.param['delta_capital_cons_limit']
        if self.co2_tax_efficiency is not None:
            if 'years' in self.co2_tax_efficiency:
                self.co2_tax_efficiency.index = self.co2_tax_efficiency['years']
            else:
                raise Exception(
                    'Miss a column years in CO2 tax efficiency to set index of the dataframe')

        self.co2_invest_limit = self.param['co2_invest_limit']
        # Employment rate param
        self.employment_a_param = self.param['employment_a_param']
        self.employment_power_param = self.param['employment_power_param']
        self.employment_rate_base_value = self.param['employment_rate_base_value']
        self.ref_emax_enet_constraint = self.param['ref_emax_enet_constraint']
        self.usable_capital_ref = self.param['usable_capital_ref']
        self.invest_co2_tax_in_renawables = self.param['assumptions_dict']['invest_co2_tax_in_renewables']
      
    def create_dataframe(self):
        """Create the dataframe and fill it with values at year_start"""
        default_index = np.arange(
            self.year_start, self.year_end + 1, self.time_step)
        param = self.param
        economics_df = pd.DataFrame(
            index=default_index,
            columns=list(GlossaryCore.Economics_df['dataframe_descriptor'].keys()))

        for key in economics_df.keys():
            economics_df[key] = 0
        economics_df['years'] = self.years_range
        economics_df.loc[param['year_start'],
                         'gross_output'] = self.init_gross_output
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
        self.energy_investment_wo_renewable['years'] = self.years_range

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
        # workforce_df
        workforce_df = pd.DataFrame(index=default_index, columns=['years',
                                                                  'employment_rate', 'workforce'])
        for key in workforce_df.keys():
            workforce_df[key] = 0
        workforce_df['years'] = self.years_range
        self.workforce_df = workforce_df
        #capital df
        capital_df = pd.DataFrame(index=default_index, 
                                  columns=['years','capital', 'non_energy_capital', 
                                           'energy_efficiency', 'e_max', 'usable_capital'])
        for key in capital_df.keys():
            capital_df[key] = 0
        capital_df['years'] = self.years_range
        capital_df.loc[param['year_start'], 'non_energy_capital'] = self.capital_start_ne
        self.capital_df = capital_df

        return economics_df.fillna(0.0), energy_investment.fillna(0.0),

    def set_coupling_inputs(self):
        """
        Set couplings inputs with right index, scaling... 
        """
        self.damefrac = self.inputs['damage_df']
        self.damefrac.index = self.damefrac['years'].values
        # Scale energy production
        self.scaling_factor_energy_production = self.inputs['scaling_factor_energy_production']
        self.scaling_factor_energy_investment = self.inputs['scaling_factor_energy_investment']
        self.energy_production = self.inputs['energy_production'].copy(deep=True)
        self.energy_production['Total production'] *= self.scaling_factor_energy_production
        self.co2_emissions_Gt = pd.DataFrame({'years': self.inputs['co2_emissions_Gt']['years'].values,
                                              'Total CO2 emissions': self.inputs['co2_emissions_Gt']['Total CO2 emissions'].values})
        self.co2_emissions_Gt.index = self.co2_emissions_Gt['years'].values
        self.co2_taxes = self.inputs['CO2_taxes']
        self.co2_taxes.index = self.co2_taxes['years'].values
        self.energy_production.index = self.energy_production['years'].values
        #Investment in energy
        self.share_energy_investment = pd.Series(
            self.inputs['share_energy_investment']['energy'].values / 100.0, index=self.years_range)
        self.share_non_energy_investment = pd.Series(
            self.inputs['share_non_energy_investment']['non_energy'].values / 100.0, index=self.years_range)
        self.energy_capital = self.inputs['energy_capital_df']
        self.energy_capital.index = self.energy_capital['years'].values
        # Population dataframes
        self.population_df = self.inputs['population_df']
        self.population_df.index = self.population_df['years'].values
        self.working_age_population_df = self.inputs['working_age_population_df']
        self.working_age_population_df.index = self.working_age_population_df['years'].values
        self.compute_gdp = self.inputs['compute_gdp']
        if not self.compute_gdp:
            self.gross_output_in = self.inputs['gross_output_in']
      
    def compute_employment_rate(self):
        """ 
        Compute the employment rate. based on prediction from ILO 
        We pyworld3 a recovery from 2020 crisis until 2031 where past level is reached
        For all year not in (2020,2031), value = employment_rate_base_value
        """
        year_covid = 2020
        year_end_recovery = 2031
        workforce_df = self.workforce_df
        # For all years employment_rate = base value
        workforce_df['employment_rate'] = self.employment_rate_base_value
        # Compute recovery phase
        years_recovery = np.arange(year_covid, year_end_recovery + 1)
        x_recovery = years_recovery + 1 - year_covid
        employment_rate_recovery = self.employment_a_param * \
            x_recovery**self.employment_power_param
        employment_rate_recovery_df = pd.DataFrame(
            {'years': years_recovery, 'employment_rate': employment_rate_recovery})
        employment_rate_recovery_df.index = years_recovery
        # Then replace values in original dataframe by recoveries values
        workforce_df.update(employment_rate_recovery_df)

        self.workforce_df = workforce_df
        return workforce_df

    def compute_workforce(self):
        """ Compute the workforce based on formula: 
        workforce = people in working age * employment_rate 
        inputs : - number of people in working age 
                - employment rate in %
        Output: number of working people in million of people
        """
        working_age_pop = self.working_age_population_df['population_1570']
        employment_rate = self.workforce_df['employment_rate']
        workforce = employment_rate * working_age_pop
        self.workforce_df['workforce'] = workforce
        return workforce

    def compute_productivity_growthrate(self, year: int):
        """
        A_g, Growth rate of total factor productivity.
        Returns:
            :returns: A_g(0) * exp(-Δ_a * (t-1))
        """
        t = ((year - self.year_start) / self.time_step) + 1
        productivity_gr = self.productivity_gr_start * \
            np.exp(-self.decline_rate_tfp * self.time_step * (t - 1))
        self.economics_df.loc[year, 'productivity_gr'] = productivity_gr
        return productivity_gr

    def compute_productivity(self, year: int):
        """
        productivity
        if damage_to_productivity= True add damage to the the productivity
        if  not: productivity evolves independently from other variables (except productivity growthrate)
        """
        damage_to_productivity = self.damage_to_productivity
        p_productivity = self.economics_df.at[year -
                                              self.time_step, 'productivity']
        p_productivity_gr = self.economics_df.at[year -
                                                 self.time_step, 'productivity_gr']
        damefrac = self.damefrac.at[year, 'damage_frac_output']
        if damage_to_productivity:
            productivity = ((1 - self.frac_damage_prod * damefrac) *
                            (p_productivity / (1 - (p_productivity_gr / (5 / self.time_step)))))
        else:
            productivity = (p_productivity /
                            (1 - (p_productivity_gr / (5 / self.time_step))))
        # we divide the productivity growth rate by 5/time_step because of change in time_step (as
        # advised in Traeger, 2013)
        self.economics_df.loc[year, 'productivity'] = productivity
        return productivity

    def compute_capital(self, year: int):
        """
        K(t+1), Capital for next time period, trillions $USD
        Args:
            :param capital: capital
            :param depreciation: depreciation rate
            :param investment: investment
        """
        year_start = self.year_start
        if year > self.year_end:
            pass
        elif year == self.year_start: 
            capital = self.capital_start_ne + self.energy_capital.loc[year_start, 'energy_capital']
            self.capital_df.loc[year_start, 'capital'] = capital
        else: 
            #first compute non energy capital 
            ne_investment = self.economics_df.at[year - self.time_step, 'non_energy_investment']
            ne_capital = self.capital_df.at[year - self.time_step, 'non_energy_capital']
            capital_a = ne_capital * (1 - self.depreciation_capital) ** self.time_step + \
                self.time_step * ne_investment
            #Then total capital = ne_capital + energy_capital 
            self.capital_df.loc[year, 'non_energy_capital'] = capital_a
            # Lower bound for capital
            tot_capital = capital_a + self.energy_capital.loc[year, 'energy_capital']
            self.capital_df.loc[year,'capital'] = max(tot_capital, self.lo_capital)
                                  
            return capital_a

    def compute_emax(self, year: int):
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
        ne_capital = self.capital_df.loc[year, 'non_energy_capital'] * 1e3
        # compute energy_efficiency
        energy_efficiency = cst + max_e / (1 + np.exp(-k * (year - xo)))
        # Then compute e_max
        e_max = ne_capital / (capital_utilisation_ratio * energy_efficiency)

        self.capital_df.loc[year,'energy_efficiency'] = energy_efficiency
        self.capital_df.loc[year, 'e_max'] = e_max

    def compute_usable_capital(self, year: int):
        """  Usable capital is the part of the capital stock that can be used in the production process. 
        To be usable the capital needs enough energy.
        K_u = K*(E/E_max) 
        E is energy in Twh and K is capital in trill dollars constant 2020
        Output: usable capital in trill dollars constant 2020
        """
        ne_capital = self.capital_df.loc[year, 'non_energy_capital']
        energy = self.energy_production.at[year, 'Total production']
        e_max = self.capital_df.loc[year, 'e_max']
        # compute usable capital
        usable_capital = ne_capital * (energy / e_max)
        self.capital_df.loc[year, 'usable_capital'] = usable_capital
        return usable_capital

    def compute_investment(self, year: int):
        """Compute I(t) (total Investment) and Ine(t) (Investment in non-energy sectors) in trillions $USD """
        net_output = self.economics_df.at[year, 'output_net_of_d']
        self.economics_df.loc[year, 'non_energy_investment'] = self.share_non_energy_investment[year] * net_output
        self.economics_df.loc[year, 'investment'] = self.economics_df.at[year, 'energy_investment'] +\
                                                    self.economics_df.at[year, 'non_energy_investment']

    def compute_energy_investment(self, year: int):
        """
        energy_investment(t), trillions $USD (including renewable investments)
        Share of the total output
        """
        net_output = self.economics_df.at[year, 'output_net_of_d']
        energy_investment_wo_tax = self.share_energy_investment[year] * net_output

        self.co2_emissions_Gt['Total CO2 emissions'].clip(lower=0.0, inplace=True)

        # store invests without renewable energy
        em_wo_ren = self.energy_investment_wo_renewable
        em_wo_ren.loc[year,
                      'energy_investment_wo_renewable'] = energy_investment_wo_tax * 1e3

        ren_investments = self.compute_energy_renewable_investment(year, energy_investment_wo_tax)
        energy_investment = energy_investment_wo_tax + ren_investments
        self.economics_df.loc[year,
                              ['energy_investment', 'energy_investment_wo_tax', 'energy_investment_from_tax']] = [energy_investment, energy_investment_wo_tax, ren_investments]
        self.energy_investment.loc[year,
                                   'energy_investment'] = energy_investment * 1e3 / self.scaling_factor_energy_investment  # Invest from T$ to G$ coupling variable

        return energy_investment

    def compute_energy_renewable_investment(self, year: int, energy_investment_wo_tax: float) -> float:
        """
        computes energy investment for renewable part
        for a given year: returns net CO2 emissions * CO2 taxes * a efficiency factor
        """
        if not self.invest_co2_tax_in_renawables:
            return 0
        co2_invest_limit = self.co2_invest_limit
        emissions = self.co2_emissions_Gt.at[year,
                                             'Total CO2 emissions'] * 1e9  # t CO2
        co2_taxes = self.co2_taxes.loc[year, 'CO2_tax']  # $/t
        co2_tax_eff = self.co2_tax_efficiency.at[year, 'CO2_tax_efficiency'] / 100.  # %
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

    def compute_gross_output(self, year: int):
        """ Compute the gdp 
        inputs: usable capital by year in trill $ , working population by year in million of people,
             productivity by year (no unit), alpha (between 0 and 1) 
        output: gdp in trillion dollars
        """
        alpha = self.output_alpha
        gamma = self.output_gamma
        productivity = self.economics_df.loc[year, 'productivity']
        working_pop = self.workforce_df.loc[year, 'workforce']
        capital_u = self.capital_df.loc[year, 'usable_capital']
        # If gamma == 1/2 use sqrt but same formula
        if gamma == 1 / 2:
            output = productivity * \
                (alpha * np.sqrt(capital_u) + (1 - alpha) * np.sqrt(working_pop))**2
        else:
            output = productivity * \
                (alpha * capital_u**gamma + (1 - alpha)
                 * (working_pop)**gamma)**(1 / gamma)
        self.economics_df.loc[year, 'gross_output'] = output

        return output

    def set_gross_output(self): 
        """
        Set gross output according to input
        """
        self.economics_df = self.economics_df.drop('gross_output', axis=1)
        self.economics_df = self.economics_df.merge(self.gross_output_in[['years', 'gross_output']], on = 'years', how='left').set_index(self.economics_df.index)

    def compute_output_growth(self, year: int):
        """
        Compute the output growth between year t and year t-1 
        Output growth of the WITNESS pyworld3 (computed from gross_output_ter)
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

    def compute_output_net_of_damage(self, year: int):
        """
        Output net of damages, trillions USD
        """
        damage_to_productivity = self.damage_to_productivity
        damefrac = self.damefrac.at[year, 'damage_frac_output']
        gross_output = self.economics_df.at[year,
                                            'gross_output']

        if damage_to_productivity:
            damage = 1 - ((1 - damefrac) /
                          (1 - self.frac_damage_prod * damefrac))
            output_net_of_d = (1 - damage) * gross_output
        else:
            output_net_of_d = gross_output * (1 - damefrac)
        self.economics_df.loc[year, 'output_net_of_d'] = output_net_of_d
        return output_net_of_d

    def compute_consumption(self, year: int):
        """Equation for consumption
        C, Consumption, trillions $USD
        Args:
            output: Economic output at t
            savings: Savings rate at t
        """
        net_output = self.economics_df.at[year, 'output_net_of_d']
        investment = self.economics_df.at[year, 'investment']
        consumption = net_output - investment
        # lower bound for conso
        self.economics_df.loc[year, 'consumption'] = max(
            consumption, self.lo_conso)
        return consumption

    def compute_consumption_pc(self, year: int):
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

    def compute_emax_enet_constraint(self):
        """ Equation for Emax constraint 
        """
        e_max = self.capital_df['e_max'].values
        energy = self.energy_production['Total production'].values
        self.emax_enet_constraint = - \
            (energy - e_max * self.max_capital_utilisation_ratio) / self.ref_emax_enet_constraint

    def compute_delta_capital_objective(self):
        """
        Compute delta between capital and a percentage of usable capital
        """
        ne_capital = self.capital_df['non_energy_capital'].values
        usable_capital = self.capital_df['usable_capital'].values
        ref_usable_capital = self.usable_capital_ref * self.nb_years
        self.delta_capital_objective_wo_exp_min = (self.capital_utilisation_ratio * ne_capital - usable_capital) / ref_usable_capital
        self.delta_capital_objective = compute_func_with_exp_min(self.delta_capital_objective_wo_exp_min, 1e-15)

    def compute_delta_capital_objective_with_alpha(self):
        """
        Compute delta between capital and a percentage of usable capital
        """

        self.delta_capital_objective_with_alpha = self.alpha * self.delta_capital_objective

    def compute_delta_capital_constraint(self):
        """
        Compute delta between capital and a percentage of usable capital
        """
        ne_capital = self.capital_df['non_energy_capital'].values
        usable_capital = self.capital_df['usable_capital'].values
        ref_usable_capital = self.usable_capital_ref * self.nb_years
        delta_capital = (self.capital_utilisation_ratio * ne_capital - usable_capital)
        self.delta_capital_cons = (self.delta_capital_cons_limit - np.sign(delta_capital) * np.sqrt(compute_func_with_exp_min(delta_capital**2, 1e-15))) / ref_usable_capital

    def compute_delta_capital_constraint_dc(self):
        """
        Compute delta between capital and a percentage of usable capital using the delta constraint function
        """
        ne_capital = self.capital_df['non_energy_capital'].values
        usable_capital = self.capital_df['usable_capital'].values
        ref_usable_capital = self.usable_capital_ref * self.nb_years
        self.delta_capital_cons_dc = compute_delta_constraint(value=usable_capital, goal=self.capital_utilisation_ratio * ne_capital,
                                                           tolerable_delta=self.delta_capital_cons_limit,
                                                           delta_type='hardmin', reference_value=ref_usable_capital)

    def compute_delta_capital_lin_to_quad_constraint(self):
        """
        Compute delta between capital and a percentage of usable capital using the lin_to_quad method.
        Through this implementation, there is a hard limit (quadratic increase) if:
         (usable_capital - self.capital_utilisation_ratio * ne_capital)>delta_capital_eq_cons_limit
        which corresponds to the case: energy > e_max * self.max_capital_utilisation_ratio + eps
        and a linear increase going away from zero otherwise
        """
        ne_capital = self.capital_df['non_energy_capital'].values
        usable_capital = self.capital_df['usable_capital'].values
        ref_usable_capital = self.usable_capital_ref * self.nb_years
        tolerable_delta = 0.15 * ne_capital
        self.delta_capital_lintoquad = compute_delta_constraint(value=usable_capital, goal=self.capital_utilisation_ratio * ne_capital,
                                                                tolerable_delta=tolerable_delta,
                                                                delta_type='normal', reference_value=ref_usable_capital)

    def check_values(self, inputs: dict):
        """check inputs"""
        if any(np.real(inputs['share_energy_investment']['energy']) > 100.):
            print('The share of energy investment is above 100% of total investment')

        if any(np.real(inputs['share_energy_investment']['energy']) < 0.):
            print('The share of energy investment is below 0% of total investment')

        if any(np.real(inputs['share_non_energy_investment']['non_energy']) > 100.):
            print('The share of non energy investment is above 100% of total investment')

        if any(np.real(inputs['share_non_energy_investment']['non_energy']) < 0.):
            print('The share of non energy investment is below 0% of total investment')

        if any(np.real(inputs['share_energy_investment']['energy'] + inputs['share_non_energy_investment']['non_energy']) > 100.):
            print('The share of investment (energy + non energy) exceeds 100% of total investment')

    def compute(self, inputs: dict):
        """
        Compute all models for year range
        """

        self.check_values(inputs)
        self.create_dataframe()
        self.inputs = deepcopy(inputs)
        self.set_coupling_inputs()
        # set gross ouput from input if necessary
        if not self.compute_gdp:
            self.set_gross_output()
        # Employment rate and workforce
        self.compute_employment_rate()
        self.compute_workforce()
        # YEAR START
        year_start = self.year_start
        self.compute_capital(year_start)
        self.compute_emax(year_start)
        self.compute_usable_capital(year_start)
        self.compute_output_net_of_damage(year_start)
        self.compute_energy_investment(year_start)
        self.compute_investment(year_start)
        self.compute_consumption(year_start)
        self.compute_consumption_pc(year_start)
        # for year 0 compute capital +1
        self.compute_capital(year_start +1)

        # Then iterate over years from year_start + tstep:
        for year in self.years_range[1:]:
            # First independant variables
            self.compute_productivity_growthrate(year)
            self.compute_productivity(year)
            # Then others:
            self.compute_emax(year)
            self.compute_usable_capital(year)
            if self.compute_gdp:
                self.compute_gross_output(year)
            self.compute_output_net_of_damage(year)
            self.compute_energy_investment(year)
            self.compute_investment(year)
            self.compute_consumption(year)
            self.compute_consumption_pc(year)
            # capital t+1 :
            self.compute_capital(year+1)

        for year in self.years_range:
            self.compute_output_growth(year)
        self.economics_df = self.economics_df.replace(
            [np.inf, -np.inf], np.nan)
        # Compute consumption per capita constraint
        self.compute_comsumption_pc_constraint()
        # Compute e_max net energy constraint
        self.compute_emax_enet_constraint()
        self.compute_delta_capital_objective()
        self.compute_delta_capital_objective_with_alpha()
        self.compute_delta_capital_constraint()
        self.compute_delta_capital_constraint_dc()
        self.compute_delta_capital_lin_to_quad_constraint()

        return self.economics_df.fillna(0.0), self.energy_investment.fillna(0.0), \
            self.energy_investment_wo_renewable.fillna(0.0), self.pc_consumption_constraint, self.workforce_df, \
            self.capital_df, self.emax_enet_constraint

    """-------------------Gradient functions-------------------"""

    def _null_derivative(self):
        nb_years = len(self.years_range)
        return np.zeros((nb_years, nb_years))

    def d_capital_d_user_input(self, d_investment_d_user_input):
        """ Compute derivative of capital wrt X. User should provide derivative of investment wrt X."""
        nb_years = self.nb_years
        d_capital_d_user_input = self._null_derivative()
        for i in range(nb_years):
            for j in range(0, i + 1):
                if i < nb_years - 1:
                    d_capital_d_user_input[i + 1, j] = d_capital_d_user_input[i, j] * (
                            1 - self.depreciation_capital) ** self.time_step + self.time_step * d_investment_d_user_input[i, j]
        return d_capital_d_user_input

    def d_emax_constraint_d_user_input(self, d_capital_d_user_input):
        """
        Derivative of emax constraint wrt X.
        User should provide derivivative of capital wrt X.

        e_max = capital / (capital_utilisation_ratio * energy_efficiency) * 1000
        """
        energy_efficiency = self.capital_df['energy_efficiency'].values

        d_emax_d_capital = np.diag(1e3 / (self.capital_utilisation_ratio * energy_efficiency))
        d_emax_d_user_input = d_emax_d_capital @ d_capital_d_user_input
        d_emax_constraint_d_user_input__normalized = d_emax_d_user_input * self.max_capital_utilisation_ratio / self.ref_emax_enet_constraint
        return d_emax_constraint_d_user_input__normalized

    def d_workforce_d_workagepop(self):
        """Gradient for workforce wrt working age population"""
        employment_rate = self.workforce_df['employment_rate'].values
        d_workforce_d_workagepop = np.diag(employment_rate)
        return d_workforce_d_workagepop

    def d_productivity_d_damage_frac_output(self):
        """gradient for productivity for damage_df"""
        nb_years = len(self.years_range)

        d_productivity_d_damage_frac_output = self._null_derivative()

        if self.damage_to_productivity:   # todo : maybe it can be computed more efficiently ?
            p_productivity_gr = self.economics_df['productivity_gr'].values
            p_productivity = self.economics_df['productivity'].values

            # first line stays at zero since derivatives of initial values are zero
            for i in range(1, nb_years):
                d_productivity_d_damage_frac_output[i, i] = (1 - self.frac_damage_prod * self.damefrac.at[self.years_range[i], 'damage_frac_output']) * \
                    d_productivity_d_damage_frac_output[i - 1, i] / (1 - (p_productivity_gr[i - 1] /
                                                     (5 / self.time_step))) -\
                    self.frac_damage_prod * \
                    p_productivity[i - 1] / \
                    (1 - (p_productivity_gr[i - 1] / (5 / self.time_step)))
                for j in range(1, i):
                    d_productivity_d_damage_frac_output[i, j] = (1 - self.frac_damage_prod * self.damefrac.at[self.years_range[i], 'damage_frac_output']) * \
                        d_productivity_d_damage_frac_output[i - 1, j] / \
                        (1 - (p_productivity_gr[i - 1] / (5 / self.time_step)))

        return d_productivity_d_damage_frac_output

    def d_usable_capital_d_energy(self):
        """
        Derivative of usable capital wrt energy
        usable_capital = capital * (energy / e_max)
        """
        ne_capital = self.capital_df['non_energy_capital'].values
        e_max = self.capital_df['e_max'].values
        d_usable_capital_d_energy = np.diag(ne_capital / e_max)
        return d_usable_capital_d_energy

    def d_gross_output_d_working_pop(self):
        """ Gradient for gross output wrt working pop
        gross output = productivity * (alpha * capital_u**gamma + (1-alpha)* (working_pop)**gamma)**(1/gamma) 
        """
        nb_years = len(self.years_range)

        if not self.compute_gdp:
            return self._null_derivative()

        alpha = self.output_alpha
        gamma = self.output_gamma
        doutput = np.identity(nb_years)
        working_pop = self.workforce_df['workforce'].values
        capital_u = self.capital_df['usable_capital'].values
        productivity = self.economics_df['productivity'].values
        # output = f(g(x)) with f = productivity*g**(1/gamma) a,d g= alpha * capital_u**gamma + (1-alpha)* (working_pop)**gamma
        # f'(g) = productivity*(1/gamma)*g**(1/gamma -1)
        # g'(workingpop) = (1-alpha)*gamma*workingpop**(gamma-1)
        # f'(g(x)) = f'(g)*g'(x)
        # first line stays at zero since derivatives of initial values are zero
        g = alpha * capital_u**gamma + (1 - alpha) * (working_pop)**gamma
        g_prime = (1 - alpha) * gamma * working_pop**(gamma - 1)
        f_prime = productivity * (1 / gamma) * (g** (1/gamma -1)) * g_prime
        doutput *= f_prime
        doutput[0, 0] = 0
        return doutput

    def d_net_output_d_user_input(self, d_gross_output_d_user_input):
        """derivative of net output wrt X, user should provide the derivative of gross output wrt X"""
        damefrac = self.damefrac['damage_frac_output'].values
        if self.damage_to_productivity:
            d_net_output_d_user_input = (
                1 - damefrac) / (1 - self.frac_damage_prod * damefrac) * d_gross_output_d_user_input
        else:
            d_net_output_d_user_input = (1 - damefrac) * d_gross_output_d_user_input
        return d_net_output_d_user_input

    def _d_energy_investment_d_share_energy_investement(self): # GOOD
        """
        energy_investment(t), trillions $USD (including renewable investments)
        Share of the total output
        """

        net_output = self.economics_df['output_net_of_d'].values
        energy_investment_wo_tax = self.share_energy_investment.values * net_output
        d_energy_investment_wo_tax_d_share_energy_investment = np.diag(net_output) / 100.
        d_energy_investment_wo_renewable_d_share_energy_investment = d_energy_investment_wo_tax_d_share_energy_investment * 1e3

        self.co2_emissions_Gt['Total CO2 emissions'].clip(
            lower=0.0, inplace=True)

        d_ren_investments_d_share_energy_investment = self._d_ren_investments_d_energy_investment_wo_tax(
            energy_investment_wo_tax, d_energy_investment_wo_tax_d_share_energy_investment)
        d_energy_investment_d_share_energy_investment = d_energy_investment_wo_tax_d_share_energy_investment + \
                                                       d_ren_investments_d_share_energy_investment

        return d_energy_investment_d_share_energy_investment, d_energy_investment_wo_renewable_d_share_energy_investment

    def d_investment_d_share_energy_investment(self):
        """Derivative of investment wrt share energy investment"""
        d_energy_investment_d_share_energy_investment, d_energy_investment_wo_renewable_d_share_energy_investment = \
            self._d_energy_investment_d_share_energy_investement()

        d_non_energy_investment_d_share_energy_investment = self._null_derivative()
        d_investment_d_share_energy_investment = d_energy_investment_d_share_energy_investment
        return d_investment_d_share_energy_investment, d_energy_investment_d_share_energy_investment,\
               d_non_energy_investment_d_share_energy_investment, d_energy_investment_wo_renewable_d_share_energy_investment

    def _d_ren_investments_d_energy_investment_wo_tax(self, energy_investment_wo_tax, denergy_investment_wo_tax):
        """
        computes gradients for energy investment for renewable part by energy_investment_wo_tax
        for a given year: returns net CO2 emissions * CO2 taxes * a efficiency factor
        """
        co2_invest_limit = self.co2_invest_limit
        # t CO2
        emissions = self.co2_emissions_Gt['Total CO2 emissions'].values * 1e9
        co2_taxes = self.co2_taxes['CO2_tax'].values  # $/t
        co2_tax_eff = self.co2_tax_efficiency['CO2_tax_efficiency'].values / 100.  # %

        ren_investments = emissions * co2_taxes * co2_tax_eff / 1e12  # T$

        nb_years = len(self.years_range)
        # derivative matrix initialization
        dren_investments = self._null_derivative()
        if self.invest_co2_tax_in_renawables:
            for i in range(nb_years):
                # if emissions is zero the right gradient (positive) is not zero but the left gradient is zero
                # when complex step we add ren_invest with the complex step and it is
                # not good
                if ren_investments[i].real == 0.0:
                    ren_investments[i] = 0.0

                # Saturation of renewable invest at n * invest wo tax with n ->
                # co2_invest_limit entry parameter
                if ren_investments[i] > co2_invest_limit * energy_investment_wo_tax[i] and ren_investments[i] != 0.0:
                    u = co2_invest_limit * energy_investment_wo_tax[i] / 10.0
                    u_prime = co2_invest_limit * \
                        denergy_investment_wo_tax[i] / 10.0
                    v = 9.0 + np.exp(- co2_invest_limit *
                                     energy_investment_wo_tax[i] / ren_investments[i])
                    v_prime = (- co2_invest_limit *
                               denergy_investment_wo_tax[i] / ren_investments[i]) * (v - 9.0)

                    dren_investments[i] = u_prime * v + v_prime * u

        return dren_investments

    def d_energy_investment_d_co2_tax(self):
        """derivative of energy investment wrt co2 tax"""
        net_output = self.economics_df['output_net_of_d'].values
        energy_investment_wo_tax = self.share_energy_investment.values * net_output
        dren_investments = self.d_renewable_investments_d_co2_tax(
            energy_investment_wo_tax)
        denergy_investment = dren_investments

        return denergy_investment

    def d_renewable_investments_d_co2_tax(self, energy_investment_wo_tax):
        """derivative of renewable investment wrt co2 tax"""
        nb_years = len(self.years_range)
        co2_invest_limit = self.co2_invest_limit
        # t CO2
        emissions = self.co2_emissions_Gt['Total CO2 emissions'].values * 1e9
        co2_taxes = self.co2_taxes['CO2_tax'].values  # $/t
        co2_tax_eff = self.co2_tax_efficiency['CO2_tax_efficiency'].values / 100.  # %

        ren_investments = emissions * co2_taxes * co2_tax_eff / 1e12  # T$
        d_ren_investments_dco2_taxes = np.diag(emissions * co2_tax_eff / 1e12)

        # derivative matrix initialization
        dren_investments = self._null_derivative()
        if self.invest_co2_tax_in_renawables:
            for i in range(nb_years):

                # if emissions is zero the right gradient (positive) is not zero but the left gradient is zero
                # when complex step we add ren_invest with the complex step and it
                # is not good
                if ren_investments[i].real == 0.0:
                    ren_investments[i] = 0.0
                    d_ren_investments_dco2_taxes[i] = np.zeros(nb_years)

                dren_investments[i] = d_ren_investments_dco2_taxes[i]

                # Saturation of renewable invest at n * invest wo tax with n ->
                # co2_invest_limit entry parameter
                if ren_investments[i] > co2_invest_limit * energy_investment_wo_tax[i] and ren_investments[i] != 0.0:
                    v = np.exp(- co2_invest_limit *
                               energy_investment_wo_tax[i] / ren_investments[i])
                    v_prime = (d_ren_investments_dco2_taxes[i] * co2_invest_limit *
                               energy_investment_wo_tax[i] / (ren_investments[i]**2))

                    dren_investments[i] = co2_invest_limit * \
                        energy_investment_wo_tax[i] / 10.0 * v_prime * v

        return dren_investments

    def d_investment_d_user_input(self, d_net_output_d_user_input):
        """derivative of investment wrt X, user should provide the derivative of net output wrt X"""
        years = self.years_range
        nb_years = len(years)
        d_energy_investment_d_user_input = self.share_energy_investment.values * d_net_output_d_user_input
        # Saturation of renewable invest at n * invest wo tax with n ->
        # co2_invest_limit entry parameter
        for i in range(nb_years):
            emissions = self.co2_emissions_Gt.at[years[i], 'Total CO2 emissions']
            co2_taxes = self.co2_taxes.at[years[i], 'CO2_tax']
            co2_tax_eff = self.co2_tax_efficiency.at[years[i],
                                                     'CO2_tax_efficiency']
            energy_investment_wo_tax = self.economics_df.at[years[i],
                                                            'energy_investment_wo_tax']
            net_output = self.economics_df.at[years[i], 'output_net_of_d']
            ren_investments = emissions * 1e9 * co2_taxes * co2_tax_eff / 100 / 1e12  if self.invest_co2_tax_in_renawables else 0  # T$
            for j in range(0, i + 1):
                if ren_investments > self.co2_invest_limit * energy_investment_wo_tax:

                    d_energy_investment_d_user_input[i, j] += self.co2_invest_limit * self.share_energy_investment[years[i]] * d_net_output_d_user_input[i, j] * 9 / 10 \
                                                + self.co2_invest_limit * self.share_energy_investment[years[i]] * d_net_output_d_user_input[i, j] / 10 * np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments) \
                                                + self.co2_invest_limit * self.share_energy_investment[years[i]] * net_output / 10 * (-1) * self.co2_invest_limit * self.share_energy_investment[years[i]] * d_net_output_d_user_input[i, j] / ren_investments \
                                                * np.exp(- self.co2_invest_limit * energy_investment_wo_tax / ren_investments)

        d_non_energy_investment_d_user_input = np.diag(self.share_non_energy_investment.values) @ d_net_output_d_user_input

        d_investment_d_user_input = d_energy_investment_d_user_input + d_non_energy_investment_d_user_input

        return d_energy_investment_d_user_input, d_investment_d_user_input, d_non_energy_investment_d_user_input

    def d_consumption_d_user_input(self, d_net_output_d_user_input, d_investment_d_user_input):
        """derivative of consumption wrt user input"""
        consumption = self.economics_df['consumption'].values
        dconsumption = d_net_output_d_user_input - d_investment_d_user_input
        # find index where lower bound reached
        theyears = np.where(consumption == self.lo_conso)[0]
        # Then for these years derivative = 0
        dconsumption[theyears] = 0
        return dconsumption

    def d_consumption_per_capita_d_user_input(self, d_consumption_d_user_input):
        """
        derivative of consumption per capita wrt user input

        consumption per capita = consumption / population * 1000
        """
        pc_consumption = self.economics_df['pc_consumption'].values

        d_consumption_per_capita_d_consumption = np.diag(1 / self.population_df['population'].values * 1000)
        d_consumption_per_capita_d_user_input = d_consumption_per_capita_d_consumption @ d_consumption_d_user_input
        # find index where lower bound reached and set it to zero
        theyears = np.where(pc_consumption == self.lo_per_capita_conso)[0]
        d_consumption_per_capita_d_user_input[theyears] = 0
        return d_consumption_per_capita_d_user_input

    def d_consumption_pc_d_population(self):
        """derivative of pc_consumption wrt population
        consumption_pc = consumption / population * 1000

        # Lower bound for pc conso
        self.economics_df.loc[year, 'pc_consumption'] = max(
            consumption_pc, self.lo_per_capita_conso)
        """
        consumption_pc = self.economics_df['pc_consumption'].values
        consumption = self.economics_df['consumption'].values
        population = self.population_df['population'].values

        d_consumption_pc_d_population = np.diag( - consumption * 1000 / population ** 2)
        # find index where lower bound reached and set them to 0
        theyears = np.where(consumption_pc == self.lo_per_capita_conso)[0]
        d_consumption_pc_d_population[theyears] = 0
        return d_consumption_pc_d_population

    def d_gross_output_d_damage_frac_output(self):
        """derivative of gross output wrt damage frac output"""
        if not self.compute_gdp or not self.damage_to_productivity:
            return self._null_derivative()

        d_productivity_d_damage_frac_output = self.d_productivity_d_damage_frac_output()

        working_pop = self.workforce_df['workforce'].values
        capital_u = self.capital_df['usable_capital'].values

        d_gross_output_d_productivity = np.diag(
            (self.output_alpha * capital_u ** self.output_gamma + (1 - self.output_alpha)
             * working_pop ** self.output_gamma) ** (1 / self.output_gamma))

        d_gross_output_d_productivity[0, 0] = 0  # at year start gross output is an input
        d_gross_output_d_damage_frac_output = d_gross_output_d_productivity @ d_productivity_d_damage_frac_output
        return d_gross_output_d_damage_frac_output

    def d_net_output_d_damage_frac_output(self, d_gross_output_d_damage_frac_output):
        """derivative of net output wrt damage frac output #todo: refactor!!!"""
        nb_years = len(self.years_range)
        d_net_output_d_damage_frac_output = self._null_derivative()
        for i in range(nb_years):
            gross_output = self.economics_df.at[self.years_range[i], 'gross_output']
            damage_frac_output = self.damefrac.at[self.years_range[i], 'damage_frac_output']
            for j in range(0, i + 1):
                if i == j:
                    if self.damage_to_productivity:
                        d_net_output_d_damage_frac_output[i, j] = (self.frac_damage_prod - 1) / ((self.frac_damage_prod * damage_frac_output - 1)**2) * gross_output + \
                                            (1 - damage_frac_output) / (1 - self.frac_damage_prod *
                                              damage_frac_output) * d_gross_output_d_damage_frac_output[i, j]
                    else:
                        d_net_output_d_damage_frac_output[i, j] = - gross_output + \
                                            (1 - damage_frac_output) * d_gross_output_d_damage_frac_output[i, j]
                else:
                    if self.damage_to_productivity:
                        d_net_output_d_damage_frac_output[i, j] = (
                            1 - damage_frac_output) / (1 - self.frac_damage_prod * damage_frac_output) * d_gross_output_d_damage_frac_output[i, j]
                    else:
                        d_net_output_d_damage_frac_output[i, j] = (
                            1 - damage_frac_output) * d_gross_output_d_damage_frac_output[i, j]
        return d_net_output_d_damage_frac_output

    def d_gross_output_d_energy(self):
        """derivative of gross output wrt energy"""

        d_usable_capital_d_energy = self.d_usable_capital_d_energy()
        if not self.compute_gdp:
            return self._null_derivative()

        years = self.years_range
        nb_years = len(years)
        alpha = self.output_alpha
        gamma = self.output_gamma
        d_output_d_cap = np.identity(nb_years)
        working_pop = self.workforce_df['workforce'].values
        capital_u = self.capital_df['usable_capital'].values
        productivity = self.economics_df['productivity'].values
        # Derivative of output wrt capital
        # output = f(g(x)) with f = productivity*g**(1/gamma) a,d g= alpha * capital_u**gamma + (1-alpha)* (working_pop)**gamma
        # f'(g) = productivity*(1/gamma)*g**(1/gamma -1)
        # g'(capital) = alpha*gamma*capital**(gamma-1)
        # f'(g(x)) = f'(g)*g'(x)
        g = alpha * capital_u**gamma + (1 - alpha) * (working_pop)**gamma
        g_prime = alpha * gamma * capital_u**(gamma - 1)
        f_prime = productivity * (1 / gamma) * (g** (1/gamma -1)) * g_prime
        d_output_d_cap *= f_prime
        # at zero gross output is an input
        d_output_d_cap[0, 0] = 0
        # Then doutput = doutput_d_prod * dproductivity
        doutput = np.dot(d_usable_capital_d_energy, d_output_d_cap)
        return doutput

    def d_investment_d_co2emissions(self):
        """derivative of investments wrt co2 emissions"""
        nb_years = len(self.years_range)
        co2_invest_limit = self.co2_invest_limit
        # t CO2
        emissions = self.co2_emissions_Gt['Total CO2 emissions'].values * 1e9
        co2_taxes = self.co2_taxes['CO2_tax'].values  # $/t
        co2_tax_eff = self.co2_tax_efficiency['CO2_tax_efficiency'].values / 100.  # %
        energy_investment_wo_tax = self.economics_df['energy_investment_wo_tax'].values

        ren_investments = np.zeros(nb_years)
        dren_investments = self._null_derivative()

        if self.invest_co2_tax_in_renawables:
            ren_investments = emissions * co2_taxes * co2_tax_eff / 1e12  # T$
            dren_investments = np.identity(nb_years) * co2_taxes * co2_tax_eff / 1e12 * 1e9

            for i in range(nb_years):
                if ren_investments[i] > co2_invest_limit * energy_investment_wo_tax[i] and ren_investments[i] != 0.0:
                    g_prime = co2_taxes[i] * co2_tax_eff[i] / 1e12 * 1e9
                    f_prime = g_prime * co2_invest_limit * energy_investment_wo_tax[i] / 10.0 * (co2_invest_limit * energy_investment_wo_tax[i] / ren_investments[i]**2) *\
                        np.exp(- co2_invest_limit *
                               energy_investment_wo_tax[i] / ren_investments[i])
                    dren_investments[i, i] = f_prime
                if ren_investments[i].real == 0.0:
                    dren_investments[i, i] = 0.0

        denergy_invest = dren_investments
        dinvestment = denergy_invest
        return denergy_invest, dinvestment

    def d_investment_d_share_investment_non_energy(self):
        """Derivative of Investment wrt to share investments non-energy"""
        net_output = self.economics_df['output_net_of_d'].values
        d_non_energy_investment_d_share_investment_non_energy = np.diag(net_output / 100.0)
        d_investment_d_share_investment_non_energy = d_non_energy_investment_d_share_investment_non_energy

        return d_investment_d_share_investment_non_energy, d_non_energy_investment_d_share_investment_non_energy

    def d_net_output_d_share_energy_invest(self):
        """derivative of net output wrt share energy_invest"""
        return self._null_derivative()

    """-------------------END of Gradient functions-------------------"""
