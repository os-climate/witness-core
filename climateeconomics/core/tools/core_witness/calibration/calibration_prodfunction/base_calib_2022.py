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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from scipy.interpolate import interp1d
from climateeconomics.core.core_witness.calibration.base_optim import BaseOptim
from sos_trades_core.tools.grad_solvers.validgrad.FDGradient import FDGradient
from sos_trades_core.execution_engine.func_manager.func_manager import FunctionManager

from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart,\
    InstanciatedSeries


class BaseCalibBis(BaseOptim):
    def __init__(self, optim_name=__name__):
        BaseOptim.__init__(self, optim_name)
        self.base_path = os.path.dirname(os.path.realpath(__file__))
        self.energy_df = None
        self.pib_base_df = None
        self.population_df = None
        self.capital_df = None
        self.reload_inputs()
        self.set_data()

    def reload_inputs(self):
        energy_df = pd.read_csv(os.path.join(
            self.base_path, 'data', 'net_energy_tfc.csv'))
        self.energy_df = energy_df.set_index(energy_df['Year'])
        pib_base_df = pd.read_csv(os.path.join(
            self.base_path, 'data', 'gdp_ppp_for_calib.csv'))
        self.pib_base_df = pib_base_df.set_index(pib_base_df['year'])
        population_df = pd.read_csv(os.path.join(
            self.base_path, 'data', 'workforce_data.csv'))
        #For workforce data. we used data from world bank from 1990 and linear regression before. 
        #workforce = 38.73403992*year - 74721.586 
        self.population_df = population_df.set_index(population_df['year'])
        capital_df = pd.read_csv(os.path.join(
            self.base_path, 'data', 'capital_2020.csv'))
        self.capital_df = capital_df.set_index(capital_df['year'])

    def set_data(self, year_start = 1965, year_end = 2019):
        self.year_start = year_start
        self.year_end = year_end
        self.pib_year_start = self.pib_base_df.loc[self.year_start, 'GDP']/1e12  # Value at t=0 of pib_base
        self.year_range = np.arange(self.year_start, self.year_end + 1)
        data = np.zeros(len(self.year_range))
        self.productivity_df = pd.DataFrame({'year': self.year_range, 'productivity': data,
                                             'productivity_gr': data}, index=self.year_range)
        self.pib_df = pd.DataFrame(
            {'year': self.year_range, 'output': data}, index=self.year_range)

    def optim_pib(self, x, bounds):
        self.database = {tuple(x): {}}
        self.current_iter = -1
        self._log_iteration_vector(x)
        x_opt, f, d = fmin_l_bfgs_b(self.eval_all, x, fprime=self.FD_compute_all, bounds=bounds, maxfun=10000, approx_grad=0,
                                    maxiter=1000, m=len(x), iprint=1, pgtol=1.e-9, factr=1., maxls=2 * len(x), callback=self._log_iteration_vector)
        self.f = f
        self.d = d
        return x_opt

    def eval_all(self, x):
        """
        Base eval all 
        """
        pass
    
    def compute_mod_func(self):
        self.delta_pib = self.comp_delta_pib()
        #self.delta_var = self.comp_delta_var()
        self.delta_sum = self.delta_pib
        func_manager = FunctionManager()
        #-- reference square due to delta**2
        func_manager.add_function('cst_delta_pib', np.array(
            self.delta_pib), FunctionManager.INEQ_CONSTRAINT, weight=1.)
        func_manager.build_aggregated_functions(eps=1e-3)
        self.mod_func = func_manager.mod_obj
        return self.mod_func

    def FD_compute_all(self, x):
        grad_eval = FDGradient(1j, self.eval_all, fd_step=1.e-12)
        grad_eval.set_multi_proc(False)
        outputs_grad = grad_eval.grad_f(x)
        return outputs_grad

    def comp_delta_pib(self):
        """ Compute (y_ref - y_comp)^2 and returns a series
        Inputs: 2 dataframes"""
        pib_base = self.pib_base_df
        pib_base_df = pib_base.loc[(pib_base['year'] >= self.year_start) & (
            pib_base['year'] <= self.year_end)]
        delta = (pib_base_df['GDP'] / 1e12 -
                 self.pib_df['output']) / (pib_base_df['GDP'] / 1e12)
        for year in np.arange(1975, 1986):
            delta[year] = delta[year]*3
#         for year in np.arange(1990, 1994):
#             delta[year] = delta[year]/2
        for year in np.arange(2007, 2011): 
            delta[year] = delta[year]*10
        for year in np.arange(2011,2018):
            delta[year] = delta[year]*7
        delta[2019] = delta[2019]*20
        delta[2018] = delta[2018]*10    
#         for year in np.arange(self.year_start, 1991):
#             delta[year] = 0.8*delta[year]
        absdelta = np.sign(delta) * delta
        return absdelta

    def compute_estimated_pib(self):
        pass
    
    def compute_usable_capital(self, capital_df, energy_df):
        """
        Usable capital is the part of the capital stock that can be used in the production process. 
        To be usable the capital needs enough energy.
        K_u = K*(E/E_max) 
        E_max = K/(0.8*energy_efficiency(year)
        E in Twh and K in trill dollars constant 2020
        0.8 is the capital utilization rate
        """
        capital = capital_df['capital']
        energy = energy_df['Energy']
        #compute e_max
        e_max = self.compute_e_max(capital_df)
        #compute usable capital
        usable_capital = capital * (energy / e_max)
        self.capital_df['usable_capital'] = usable_capital
        #self.plot_usable_capital()
        return usable_capital
    
    def compute_e_max(self, capital_df):
        """
        E_max is the maximum energy capital can use to produce output
        E_max = K/(0.8*energy_efficiency(year)
        energy_efficiency = 1+ max/(1+exp(-k(x-x0)))
        energy_efficiency is a logistic function because it represent technological progress
        k = 0.073821941, cst = 1, xo = 2013.402526, max = 2.042246788
        """
        k = 0.0508535660058912
        cst = 0.983523274104558
        xo = 2012.83269463152
        max = 3.51647672511314
        capital_utilisation_ratio = 0.8
        #Convert capital in billion
        capital = capital_df['capital']*1e3
        years = pd.Series(self.year_range)
        years.index = years
        #compute energy_efficiency
        energy_efficiency = cst + max/(1+np.exp(-k*(years-xo)))     
        #Then compute e_max
        e_max = capital/ (capital_utilisation_ratio * energy_efficiency) 
        #store it in a nice df 
        self.e_max_df = pd.DataFrame({'year': years, 'energy_efficiency': energy_efficiency,
                                      'e_max': e_max})
        #self.plot_e_max()
        return e_max
    
    def compute_emax_year (self, year,capital_df):
        """ Same as above but compute for only one specific year
        """
        k = 0.0508535660058912
        cst = 0.983523274104558
        xo = 2012.83269463152
        max = 3.51647672511314
        capital_utilisation_ratio = 0.8
        #Convert capital in billion
        capital = capital_df.loc[year, 'capital']*1e3
        #compute energy_efficiency
        energy_efficiency = cst + max/(1+np.exp(-k*(year-xo)))
        #Then compute e_max
        e_max = capital/ (capital_utilisation_ratio * energy_efficiency) 
        
        self.e_max_df.loc[year, 'energy_efficiency'] = energy_efficiency
        self.e_max_df.loc[year, 'e_max'] = e_max
    
    def compute_usable_capital_year(self, year,capital_df, energy_df):
        """ Same as above but yearly computation 
        """
        capital = capital_df.loc[year, 'capital']
        energy = energy_df.loc[year, 'Energy']
        #compute e_max
        self.compute_emax_year(year, capital_df)
        e_max = self.e_max_df.loc[year,'e_max']
        #compute usable capital
        usable_capital = capital * (energy / e_max)
        self.capital_df.loc[year, 'usable_capital'] = usable_capital
        #self.plot_usable_capital()
        return usable_capital
                    
        
    def compute_productivity_growth_rate(self, year, year_start):
        """Productivity growth rate function from DICE, Nordhaus
        input: decline rate tfp, productivity growth at year start
        """
        t = ((year - year_start) / 1) + 1
        productivity_gr = self.productivity_gr_start * \
            np.exp(-self.decline_rate_tfp * (t - 1))
        return productivity_gr

    def compute_productivity(self, year):
        """Productivity function from DICE, Nordhaus
        inputs: dataframe productivity 
        """
        p_productivity = self.productivity_df.loc[year - 1, 'productivity']
        p_productivity_gr = self.productivity_df.loc[year -
                                                     1, 'productivity_gr']
        productivity = (p_productivity / (1 - p_productivity_gr))
        return productivity
    
    def set_data_projection(self, scenario = "high"):
        brut_net = 1/1.45
        energy_level_dict = {'high': 1, 'low': 0.2}
        #prepare energy df  
        energy_outlook = pd.DataFrame({
            'year': [2010, 2017, 2018, 2025, 2030, 2035, 2040, 2050],
            'energy': [149483.879, 162784.8774, 166468.5636, 180707.2889, 189693.2084, 197841.8842, 206120.1182, 220000]})
        f2 = interp1d(energy_outlook['year'], energy_outlook['energy'])
        #Find values for 2020, 2050 and concat dfs 
        energy_supply = f2(np.arange(2020, 2051))
        energy_supply_values = energy_supply * brut_net * energy_level_dict[scenario]
        energy_supply_df_base = self.energy_df.loc[self.energy_df['Year']<2020]
        energy_supply_df_base =  energy_supply_df_base.rename(columns = {'Year': 'year'})
        energy_supply_df_future = pd.DataFrame({'year': self.years_proj, 'Energy': energy_supply_values})
        energy_supply_df_future.index = self.years_proj
        self.energy_df = pd.concat([energy_supply_df_base,energy_supply_df_future])
        #Add 2020 crisis
        self.energy_df.loc[2020, 'Energy'] = self.energy_df.loc[2020, 'Energy']*0.96
        self.energy_df.loc[2021, 'Energy'] = self.energy_df.loc[2019, 'Energy']
        #Population
        population_df = pd.read_csv(os.path.join(
            self.base_path, 'data', 'ihmefutureworkforce.csv'))
        #INSERT HOW IT WAS COMPUTED 
        self.population_df = population_df.set_index(population_df['year'])
        #Prepare pib df 
        data_zeros = np.zeros(len(self.years_proj))
        pib_future_df = pd.DataFrame({'year': self.years_proj, 
                                         'output': data_zeros}) 
        pib_future_df.index = self.years_proj
        pib_base_concat = self.pib_base_df.loc[self.pib_base_df['year']<2020] 
        pib_base_concat = pib_base_concat.rename(columns={'GDP' : 'output'})
        pib_base_concat['output'] = pib_base_concat['output']/1e12
        self.pib_df = pd.concat([pib_base_concat, pib_future_df])
        #emax
        self.e_max_df = pd.DataFrame({'year': self.years_proj, 'energy_efficiency': data_zeros,
                                      'e_max': data_zeros })
        #capital df
        capital_df = self.capital_df.loc[self.capital_df['year']<2020]
        capital_future_df = pd.DataFrame({'year': self.years_proj, 
                                         'capital': data_zeros, 'usable_capital': data_zeros}) 
        capital_future_df.index = self.years_proj
        self.capital_df = pd.concat([capital_df, capital_future_df])
        productivity_future_df = pd.DataFrame({'year': self.years_proj, 'productivity': data_zeros,
                                               'productivity_gr': data_zeros})
        productivity_future_df.index = self.years_proj
        productivity_df = self.productivity_df
        self.productivity_df = pd.concat([productivity_df, productivity_future_df])
        self.e_max_df = pd.DataFrame({'year': self.years_proj, 'energy_efficiency': data_zeros,
                                               'e_max': data_zeros })
        self.e_max_df.index = self.years_proj
        
    def eval_projection(self,year_end, x, scenario, origin_year_start = 1965):
        """ Compute pib projection using chosen x
        """
        years_tot = np.arange(origin_year_start, year_end+1)
        years_proj = np.arange(2020, year_end+1)
        self.years_proj = years_proj
        self.years_tot = years_tot
        self.productivity_gr_start = x[0]
        self.productivity_start = x[1]
        self.decline_rate_tfp = x[2]
        self.alpha = x[3]  
        
        self.set_data_projection(scenario)
        
        self.productivity_df.loc[origin_year_start,
                                 'productivity_gr'] = self.productivity_gr_start
        self.productivity_df.loc[origin_year_start,
                                 'productivity'] = self.productivity_start
        for year in years_tot[1:]:
            self.productivity_df.loc[year, 'productivity_gr'] = self.compute_productivity_growth_rate(
                year, origin_year_start)
            self.productivity_df.loc[year, 'productivity'] = self.compute_productivity(year)
        working_pop = self.population_df
        #And compute pib and capital 
        for year in years_proj:
            #compute capital using data from year-1
            self.capital_df.loc[year, 'capital'] = self.compute_capital(year, self.pib_df)
            #Compute capital_u 
            self.compute_usable_capital_year(year,self. capital_df, self.energy_df)
            capital_u = self.capital_df['usable_capital']
            #compute gdp
            self.pib_df.loc[year, 'output'] = self.compute_estimated_pib(x, year, self.productivity_df, working_pop, capital_u)
            if year == 2020: 
                self.pib_df.loc[year, "output"] = 130.187
        self.plot_pib()
        self.plot_usable_capital()
        self.plot_productivity()
        self.plot_energy()
        self.plot_e_max()
        return self.pib_df 
        
    def compute_capital(self, year, pib_df, depreciation_rate = 0.07, saving_rate = 0.27):
        """ capital = pib*saving_rate + capital_before*(1- depreciation_rate)
        """
        pib = pib_df.loc[year -1, 'output']
        capital_before = self.capital_df.loc[year-1, 'capital']
        capital = pib*saving_rate + capital_before*(1- depreciation_rate)
        return capital 
            
    
    def plot_e_max(self):
        new_chart = TwoAxesInstanciatedChart('years', 'no unit', chart_name = 'e_max')
        new_chart.add_series(InstanciatedSeries(
            self.e_max_df.index.tolist(), self.e_max_df['e_max'].tolist(), 'e_max', InstanciatedSeries.LINES_DISPLAY))

        new_chart.to_plotly().show()
        
    def plot_pib(self):
        """
        Plot of Pib ref vs pib computed 
        """

        new_chart = TwoAxesInstanciatedChart('years', 'pib in trill$')
        pib_base = self.pib_base_df
        pib_base_df = pib_base.loc[(pib_base['year'] >= self.year_start) & (
            pib_base['year'] <= self.year_end)]
        new_chart.add_series(InstanciatedSeries(
            pib_base_df['year'].tolist(), (pib_base_df['GDP'] / 1e12).to_numpy().real.tolist(), 'base', InstanciatedSeries.LINES_DISPLAY))
        new_chart.add_series(InstanciatedSeries(
            self.pib_df['year'].tolist(), (self.pib_df['output']).to_numpy().real.tolist(), 'eval', InstanciatedSeries.LINES_DISPLAY))        

        new_chart.to_plotly().show()
        
    def plot_usable_capital(self):
        """
        Plot usable capital and capital
        """
        
        new_chart = TwoAxesInstanciatedChart('years', 'trill$', 
                                             chart_name = 'Capital and usable capital')

        new_chart.add_series(InstanciatedSeries(
            self.capital_df['year'].tolist(), (self.capital_df['capital']).to_numpy().real.tolist(), 'capital', InstanciatedSeries.LINES_DISPLAY))

        new_chart.add_series(InstanciatedSeries(
            self.capital_df['year'].tolist(), (self.capital_df['usable_capital']).to_numpy().real.tolist(), 'usable capital', InstanciatedSeries.LINES_DISPLAY))

        new_chart.to_plotly().show()  

    def plot_cap_u_pib(self):
        new_chart = TwoAxesInstanciatedChart('years', 'trill$', 
                                             chart_name = 'Usable capital and gdp')
        pib_base = self.pib_base_df
        pib_base_df = pib_base.loc[(pib_base['year'] >= self.year_start) & (
            pib_base['year'] <= self.year_end)]
        new_chart.add_series(InstanciatedSeries(
            self.capital_df['year'].tolist(), (self.capital_df['usable_capital']).to_numpy().real.tolist(), 'usable capital', InstanciatedSeries.LINES_DISPLAY))
        new_chart.add_series(InstanciatedSeries(
            pib_base_df['year'].tolist(), (pib_base_df['GDP'] / 1e12).to_numpy().real.tolist(), 'gdp', InstanciatedSeries.LINES_DISPLAY))
#         new_chart.add_series(InstanciatedSeries(
#             pib_base_df['year'].tolist(), (self.population_df['workforce']/10).to_numpy().real.tolist(), 'population', InstanciatedSeries.LINES_DISPLAY))
        
        
        new_chart.to_plotly().show()  
        
    def plot_productivity(self):
        new_chart = TwoAxesInstanciatedChart('years', 'no unit', chart_name = 'productivity')
        new_chart.add_series(InstanciatedSeries(
            self.productivity_df['year'].tolist(), self.productivity_df['productivity'].tolist(), 'productivity', InstanciatedSeries.LINES_DISPLAY))

        new_chart.to_plotly().show()
    
    def plot_energy(self):
        new_chart = TwoAxesInstanciatedChart('years', 'TWh', chart_name = 'Energy')
        new_chart.add_series(InstanciatedSeries(
            self.energy_df['year'].tolist(), self.energy_df['Energy'].tolist(), 'Energy', InstanciatedSeries.LINES_DISPLAY))
        new_chart.to_plotly().show()
    
    def plot_workforce(self):
        new_chart = TwoAxesInstanciatedChart('years', 'million of people', chart_name = 'workforce')
        new_chart.add_series(InstanciatedSeries(
            self.population_df['year'].tolist(), self.population_df['workforce'].tolist(), 'workforce', InstanciatedSeries.LINES_DISPLAY))
        population_df_base = pd.read_csv(os.path.join(self.base_path, 'data', 'workforce_data.csv'))
        population_df_base.set_index(population_df_base['year'])
        new_chart.add_series(InstanciatedSeries(
             population_df_base['year'].tolist(),  population_df_base['workforce'].tolist(), 'workforce data', InstanciatedSeries.LINES_DISPLAY))
        new_chart.to_plotly().show()





