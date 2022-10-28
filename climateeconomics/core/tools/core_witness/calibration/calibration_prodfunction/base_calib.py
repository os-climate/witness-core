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
from logging import Logger, getLogger, Formatter, StreamHandler, FileHandler, basicConfig, DEBUG, INFO
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart,\
    InstanciatedSeries


class BaseCalib(BaseOptim):
    def __init__(self, optim_name=__name__):
        BaseOptim.__init__(self, optim_name)
        self.base_path = os.path.dirname(os.path.realpath(__file__))
        self.energy_df = None
        self.pib_base_df = None
        self.population_df = None
        self.capital_df = None
        self.reload_inputs()
        self.database = {}
        self.current_iter = None
        self.set_data()

    def reload_inputs(self):
        energy_df = pd.read_csv(os.path.join(
            self.base_path, 'data', 'net_energy.csv'))
        self.energy_df = energy_df.set_index(energy_df['Year'])
        pib_base_df = pd.read_csv(os.path.join(
            self.base_path, 'data', 'gdp_ppp.csv'))
        self.pib_base_df = pib_base_df.set_index(pib_base_df['year'])
        population_df = pd.read_csv(os.path.join(
            self.base_path, 'data', 'population_df.csv'))
        self.population_df = population_df.set_index(population_df['year'])
        capital_df = pd.read_csv(os.path.join(
            self.base_path, 'data', 'capital_2020.csv'))
        self.capital_df = capital_df.set_index(capital_df['year'])

    def set_data(self, year_start = 1980, year_end = 2019):
        self.year_start = year_start
        self.year_end = year_end
        self.pib_year_start = self.pib_base_df.loc[self.year_start, 'GDP']/1e12  # Value at t=0 of pib_base
        self.year_range = np.arange(self.year_start, self.year_end + 1)
        data = np.zeros(len(self.year_range))
        self.productivity_df = pd.DataFrame({'year': self.year_range, 'productivity': data,
                                             'productivity_gr': data}, index=self.year_range)
        self.pib_df = pd.DataFrame(
            {'year': self.year_range, 'output': data}, index=self.year_range)
        self.energy_intens = pd.Series(data, index=self.year_range)

    def optim_pib(self, x, bounds):
        self.database = {tuple(x): {}}
        self.current_iter = -1
        self._log_iteration_vector(x)
        x_opt, f, d = fmin_l_bfgs_b(self.eval_all, x, fprime=self.FD_compute_all, bounds=bounds, maxfun=10000, approx_grad=0,
                                    maxiter=1000, m=len(x), iprint=1, pgtol=1.e-9, factr=1., maxls=2 * len(x), callback=self.__log_iteration_vector)
        self.f = f
        self.d = d
        return x_opt

    def _log_iteration_vector(self, xk):
        """ Callback method attach to fmin_l_bgfs that capture each vector use during optimization

        :params: xk, current optimization vector
        :type: list
        """
        self.current_iter += 1
        msg = "ite " + str(self.current_iter) + \
            " x_%i " % self.current_iter + str(xk)
        if tuple(xk) in self.database:
            inputs = self.database[tuple(xk)]
            for key, value in inputs.items():
                msg += ' ' + key + ' ' + str(value)
        self.__logger.info(msg)

    def eval_all(self, x):
        """
        Base eval all 
        """
        self.energy_factor = x[0]
        self.gr_rate_energy = x[1]
        self.productivity_start = x[2]
        self.productivity_gr_start = x[3]
        self.decline_rate_tfp = x[4]
        # Initialise everything
        # self.set_data()
        year_range = self.year_range
        self.productivity_df.loc[year_range[0],
                                 'productivity_gr'] = self.productivity_gr_start
        self.productivity_df.loc[year_range[0],
                                 'productivity'] = self.productivity_start
        self.energy_intens[year_range[0]] = self.energy_factor
        # COmpute productivity and energy intensity
        for year in year_range[1:]:
            self.productivity_df.loc[year, 'productivity_gr'] = self.compute_productivity_growth_rate(
                year)
            self.productivity_df.loc[year, 'productivity'] = self.compute_productivity(
                year)
            self.energy_intens[year] = self.compute_energy_intensity(year)
        # COmpute pib
        for year in year_range:
            self.pib_df.loc[year, 'output'] = self.compute_estimated_pib(
                year, x)
        # Compute delta
        self.compute_mod_func()
        self.database[tuple(x)] = {'cst_delta_pib': self.mod_func}
        return self.mod_func

    def compute_mod_func(self):
        self.delta_pib = self.comp_delta_pib()
        self.delta_var = self.comp_delta_var()
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
        # add harder contraints on some points
#         delta.loc[pib_base['years'] ==
#                   1965] = delta.loc[pib_base['years'] == 1965] * 3.
#         delta.loc[pib_base['years']== 1995] = delta.loc[pib_base['years']== 1995]*3.
#         delta.loc[pib_base['year'] ==
#                   2008] = delta.loc[pib_base['year'] == 2008] * 2
#         delta.loc[pib_base['year'] ==
#                   2009] = delta.loc[pib_base['year'] == 2009] * 2
#         delta.loc[pib_base['years'] ==
#                   2009] = delta.loc[pib_base['years'] == 2009] * 3.
#         delta.loc[pib_base['year'] ==
#                    2015] = delta.loc[pib_base['year'] == 2015] * 3.
        for year in np.arange(self.year_start, 1993):
            delta[year] = delta[year]/2
        absdelta = np.sign(delta) * delta
        return absdelta

    def comp_delta_var(self):
        """Compute the difference in variation"""
        var_serie = pd.Series(
            np.zeros(len(self.year_range)), index=self.year_range)
        pib_base = self.pib_base_df
        pib_df = self.pib_df
        for year in self.year_range[1:]:
            pib_base_y = pib_base.loc[year, 'GDP'] / 1e12
            p_pib_base = pib_base.loc[year - 1, 'GDP'] / 1e12
            pib_y = pib_df.loc[year, 'output']
            p_pib = pib_df.loc[year - 1, 'output']
            var_pib_base = pib_base_y - p_pib_base
            var_pib_y = pib_y - p_pib
            delta_var = var_pib_base - var_pib_y
            var_serie.loc[year] = delta_var**2
        return var_serie

    def compute_estimated_pib(self):
        pass

    def compute_productivity_growth_rate(self, year):
        """Productivity growth rate function from DICE, Nordhaus
        input: decline rate tfp, productivity growth at year start
        """
        t = ((year - self.year_start) / 1) + 1
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
        productivity = (p_productivity / (1 - (p_productivity_gr / 5)))
        return productivity

    def compute_energy_intensity(self, year):
        energy_intens = self.energy_factor * \
            (1 + self.gr_rate_energy)**(year - self.year_range[0])
        return energy_intens
    
    def create_projection_df(self):
        ## Data for projection 
        #Energy: invented value for 2050 and IEA until 2040
        energy_outlook = pd.DataFrame({
            'years': [2010, 2017, 2018, 2025, 2030, 2035, 2040, 2050],
            'energy': [149483.879, 162784.8774, 166468.5636, 180707.2889, 189693.2084, 197841.8842, 206120.1182, 220000]})
        f2 = interp1d(energy_outlook['years'], energy_outlook['energy'])
        #Find values for 2020, 2050 and concat dfs 
        energy_supply = f2(np.arange(2020, 2051))
        energy_supply_df_future = pd.DataFrame({'Year': np.arange(2020, 2051), 'Energy': energy_supply*0.91})
        energy_supply_df_future.index = np.arange(2020, 2051)
        self.energy_df = pd.concat([ self.energy_df,energy_supply_df_future])
        #Population
        popasym = 9700.0
        population_growth = 0.134
        pop_df = pd.DataFrame({'year':np.arange(2021, self.year_end+1), 'population': 
                               np.zeros(len(np.arange(2021,self.year_end+1 )))})
        self.population_df = pd.concat([self.population_df, pop_df])
        for year in np.arange(2021, self.year_end+1):
            p_population = self.population_df.loc[year-1, 'population']
            self.population_df.loc[year, 'population'] = p_population * \
            (popasym / p_population) ** population_growth
        #Prepare capital 
        capital_future_df = pd.DataFrame({'year': np.arange(2021, self.year_end+1), 
                                         'capital': np.zeros(len(np.arange(2021,self.year_end+1 )))}) 
        capital_future_df.index = np.arange(2021,self.year_end+1 ) 
        self.capital_df = pd.concat([self.capital_df, capital_future_df])
        #Prepare pib df 
        pib_future_df = pd.DataFrame({'year': self.year_range, 
                                         'output': np.zeros(len(self.year_range))}) 
        pib_future_df.index = self.year_range 
        pib_base_concat = self.pib_base_df.loc[self.pib_base_df['year']<self.year_start] 
        pib_base_concat = pib_base_concat.rename(columns={'GDP' : 'output'})
        pib_base_concat['output'] = pib_base_concat['output']/1e12
        self.pib_df = pd.concat([pib_base_concat, pib_future_df])
            
    def projection_eval(self,x):
        self.energy_factor = x[0]
        self.decline_rate_energy = x[1]
        self.productivity_start = x[2]
        self.productivity_gr_start = x[3]
        self.decline_rate_tfp = x[4]
        self.init_gr_energy = x[10]
        self.set_data(2020, 2050)
        #create and concat df 
        self.create_projection_df()
        ### productivities
        years_tot = np.arange(1980, self.year_end+1)
        data = np.zeros(len(years_tot))
        self.productivity_df = pd.DataFrame({'year': years_tot, 'productivity': data,
                                             'productivity_gr': data}, 
                                            index =years_tot)
        self.energy_intens_df = pd.DataFrame({'year': years_tot, 'energy_intensity': data,
                                                 'energy_intens_gr': data},
                                                 index= years_tot)
        self.energy_intens_df.loc[1980, 'energy_intensity'] =  self.energy_factor
        self.energy_intens_df.loc[1980, 'energy_intens_gr'] = self.init_gr_energy
        self.productivity_df.loc[1980,
                                 'productivity_gr'] = self.productivity_gr_start
        self.productivity_df.loc[1980,
                                 'productivity'] = self.productivity_start
                                 
        for year in years_tot[1:]:
            self.energy_intens_df.loc[year,'energy_intens_gr']= self.compute_energy_intens_gr(year)
            self.productivity_df.loc[year, 'productivity_gr'] = self.compute_productivity_growth_rate(
                year)
            self.productivity_df.loc[year, 'productivity'] = self.compute_productivity(year)
            self.energy_intens_df.loc[year, 'energy_intensity'] = self.compute_energy_intensity(year)
        #COmpute PIB and capital 
        for year in self.year_range:
            pib_p = self.pib_df.loc[year-1, 'output']
            capital_p = self.capital_df.loc[year-1, 'capital']
            capital = 0.92 * capital_p + 0.25*pib_p
            self.capital_df.loc[year,'capital'] = capital
            self.pib_df.loc[year, 'output']  = self.compute_estimated_pib(year, x)
        return self.pib_df 
        
    def plot_pib_energy(self):
        """Plot of PIB eval and pib ref on left axis in M$ and Energy on right axis in TWh 
        """
        # Plot energy
        pib_df = self.pib_df
        pib_base_df = self.pib_base_df
        energy_df = self.energy_df
        fig, ax_left = plt.subplots()
        ax_right = ax_left.twinx()
        lns1 = ax_left.plot(
            pib_df['year'], pib_df['output'] * 1e3, label='pib eval')
        lns2 = ax_left.plot(
            pib_base_df['year'], pib_base_df['GDP'] / 1e9, label='pib base')
        lns3 = ax_right.plot(
            energy_df['Year'], energy_df['Energy'], label='Energy', color='green')
        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        ax_left.legend(lns, labs, loc=0)
        ax_right.set_xlabel('year')
        ax_left.set_ylabel("GDP in M$")
        ax_right.set_ylabel('Energy in TWh')

        new_chart = TwoAxesInstanciatedChart(
            abscissa_axis_name='years', primary_ordinate_axis_name='GDP in M$', secondary_ordinate_axis_name='Energy in TWh')

        new_chart.add_series(InstanciatedSeries(
            self.pib_df['year'].tolist(), (self.pib_df['output'] * 1e3).to_numpy().real.tolist(), 'pib eval', InstanciatedSeries.LINES_DISPLAY))

        new_chart.add_series(InstanciatedSeries(
            self.pib_base_df['year'].tolist(), (self.pib_base_df['GDP'] / 1e9).to_numpy().real.tolist(), 'pib base', InstanciatedSeries.LINES_DISPLAY))

        new_chart.add_series(InstanciatedSeries(
            energy_df['Year'].tolist(), energy_df['Energy'].tolist(), 'Energy', InstanciatedSeries.LINES_DISPLAY, y_axis=InstanciatedSeries.Y_AXIS_SECONDARY))

        new_chart.to_plotly().show()

    def plot_pib(self):
        """
        Plot of Pib ref vs pib computed 
        """

        new_chart = TwoAxesInstanciatedChart('years', 'pib in trill$')

        new_chart.add_series(InstanciatedSeries(
            self.pib_df['year'].tolist(), (self.pib_df['output']).to_numpy().real.tolist(), 'eval', InstanciatedSeries.LINES_DISPLAY))

        new_chart.add_series(InstanciatedSeries(
            self.pib_base_df['year'].tolist(), (self.pib_base_df['GDP'] / 1e12).to_numpy().real.tolist(), 'base', InstanciatedSeries.LINES_DISPLAY))

        new_chart.to_plotly().show()

    def plot_pib2(self):
        """
        Plot pib computed 
        """

        new_chart = TwoAxesInstanciatedChart('years', 'pib in trill$')
        
        ##Values of GDP for reference - source data world bank. For 2020 estimation with IMF "projected at �4.9 percent in 2020"
#         year_ref = [2015, 2016, 2017, 2018, 2019, 2020]
#         value_2020 = 84.848*0.961
#         values_ref = [75.958, 77.938, 80.508, 82.905, 84.848, value_2020]
        new_chart.add_series(InstanciatedSeries(
            self.pib_df['year'].tolist(), (self.pib_df['output']).to_numpy().real.tolist(), 'projection', InstanciatedSeries.LINES_DISPLAY))

        new_chart.to_plotly().show()
        
    def plot_pib_var(self):
        year_range = self.year_range
        var_pib2 = pd.Series([0] * len(np.arange(year_range[1], year_range[-1] + 1)),
                             index=np.arange(year_range[1], year_range[-1] + 1))
        var_pib_est = pd.Series([0] * len(np.arange(year_range[1], year_range[-1] + 1)),
                                index=np.arange(year_range[1], year_range[-1] + 1))
        for year in np.arange(year_range[1], year_range[-1] + 1):
            var_pib2[year] = self.pib_base_df.loc[year, 'pib'] / \
                1e9 - self.pib_base_df.loc[year - 1, 'pib'] / 1e9
            var_pib_est[year] = (
                self.pib_df.loc[year, 'output'] - self.pib_df.loc[year - 1, 'output']) * 1e3

        new_chart = TwoAxesInstanciatedChart('years', 'million $')

        new_chart.add_series(InstanciatedSeries(
            var_pib_est.index.tolist(), var_pib_est.values.tolist(), 'computed y-y_1 in million $', InstanciatedSeries.LINES_DISPLAY))

        new_chart.add_series(InstanciatedSeries(
            var_pib2.index.tolist(), var_pib2.values.tolist(), 'ref y-y-1 in million $', InstanciatedSeries.LINES_DISPLAY))

        new_chart.to_plotly().show()

    def plot_capital_input(self):

        new_chart = TwoAxesInstanciatedChart('years', 'capital in trill$')

        new_chart.add_series(InstanciatedSeries(
            self.capital_df['year'].tolist(), self.capital_df["capital"].tolist(), 'capital', InstanciatedSeries.LINES_DISPLAY))

        new_chart.to_plotly().show()

    def plot_energy_input(self):

        new_chart = TwoAxesInstanciatedChart('years', 'TWh')

        new_chart.add_series(InstanciatedSeries(
            self.energy_df['Year'].tolist(), self.energy_df["Energy"].tolist(), 'Energy', InstanciatedSeries.LINES_DISPLAY))

        new_chart.to_plotly().show()

    def plot_pib_inputs(self):
        new_chart = TwoAxesInstanciatedChart('years', '$')

        new_chart.add_series(InstanciatedSeries(
            self.pib_base_df['year'].tolist(), self.pib_base_df['GDP'].tolist(), 'pib', InstanciatedSeries.LINES_DISPLAY))
        new_chart.to_plotly().show()

    def plot_population_input(self):
        new_chart = TwoAxesInstanciatedChart('years', 'millions of people')

        new_chart.add_series(InstanciatedSeries(
            self.population_df['year'].tolist(), self.population_df['population'].tolist(), 'population', InstanciatedSeries.LINES_DISPLAY))

        new_chart.to_plotly().show()

    def plot_energy_pop_input(self):
        new_chart = TwoAxesInstanciatedChart('years', 'Twh/millions of people')
        toplot = self.energy_df['Energy'] / self.population_df["population"]
        new_chart.add_series(InstanciatedSeries(
            self.energy_df['Year'].tolist(), toplot.tolist(), 'population', InstanciatedSeries.LINES_DISPLAY))

        new_chart.to_plotly().show()

    def plot_pib_over_energy_over_pop(self):
        pib = self.pib_base_df['GDP'].loc[self.pib_base_df['year'] >= 1965]
        ratio = pib / self.energy_df['Energy'] / \
            self.population_df['population']
        print(ratio)
        new_chart = TwoAxesInstanciatedChart('years', '$/Twh/m of people')
        new_chart.add_series(InstanciatedSeries(
            self.energy_df['Year'].tolist(), ratio.tolist(), 'pib/energy/pop', InstanciatedSeries.LINES_DISPLAY))
        new_chart.to_plotly().show()

    def plot_pib_per_capita(self):
        pib = self.pib_base_df['GDP'].loc[self.pib_base_df['year'] >= 1965]
        pop = self.population_df['population'] * 1e6
        serie = pib / pop
        new_chart = TwoAxesInstanciatedChart('years', '$/per capita')
        new_chart.add_series(InstanciatedSeries(
            self.population_df['year'].tolist(), serie.tolist(), 'pib/capita', InstanciatedSeries.LINES_DISPLAY))
        new_chart.to_plotly().show()

    def plot_productivity(self):
        new_chart = TwoAxesInstanciatedChart('years', 'no unit')

        new_chart.add_series(InstanciatedSeries(
            self.productivity_df['year'].tolist(), self.productivity_df['productivity'].tolist(), 'productivity', InstanciatedSeries.LINES_DISPLAY))

        new_chart.to_plotly().show()
    
    def plot_energy_intensity_df(self):
        new_chart = TwoAxesInstanciatedChart('years', 'no unit')

        new_chart.add_series(InstanciatedSeries(
            self.energy_intens_df['year'].tolist(), self.energy_intens_df['energy_intensity'].tolist(), 'energy_intensity', InstanciatedSeries.LINES_DISPLAY))

        new_chart.to_plotly().show()

