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
from sos_trades_core.tools.grad_solvers.validgrad.FDGradient import FDGradient
from sos_trades_core.execution_engine.func_manager.func_manager import FunctionManager
from logging import Logger, getLogger, Formatter, StreamHandler, FileHandler, basicConfig, DEBUG, INFO
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart,\
    InstanciatedSeries


class BaseCalib():
    """
    CLass used for calibration of death rate. 
    """
    def __init__(self, optim_name=__name__):
        self.base_path = os.path.dirname(os.path.realpath(__file__))
        self.pib_base_df = None
        self.population_df = None
        self.reload_inputs()
        self.__logger = None
        self.__logfile_basename = optim_name
        self.__init_logger()
        self.database = {}
        self.current_iter = None
        self.set_data()

    @property
    def logger(self):
        """ Accessor on logger member variable
        """
        return self.__logger

    def __init_logger(self):
        """ Methods that initialize logging for the whole class

        Log are streamed  in two way:
        - on the standard output
        - into a log file
        """

        self.__logger = getLogger(__name__)

        # create console handler and set level to debug
        handler = StreamHandler()

        # create formatter
        formatter = Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        handler.setFormatter(formatter)
        self.__logger.addHandler(handler)

        handler = FileHandler(filename=f'{self.__logfile_basename}.log',
                              mode='w', encoding='utf-8')
        handler.setFormatter(formatter)
        self.__logger.addHandler(handler)

        self.__logger.setLevel(INFO)
        self.__logger.info('Logger initialized')

    def reload_inputs(self):
        """
        Load of inputs csv 
        """
        self.deathrate_df_full = pd.read_csv(os.path.join(
            self.base_path, 'data', 'death_rate_by_age_3_ihme.csv'))
        population_df = pd.read_csv(os.path.join(
            self.base_path, 'data', 'population_1960.csv'))
        population_df = population_df.set_index(population_df['year'])
        self.population_df = population_df
        gdp_df = pd.read_csv(os.path.join(
            self.base_path, 'data', 'gdp_ppp_for_calib.csv'))
        self.gdp_df = gdp_df.set_index(gdp_df['year'])
    
    def set_data(self, year_start = 1960, year_end = 2020):
        """
        Set all the dataframe with the right year range
        """
        self.year_start = year_start
        self.year_end = year_end
        self.year_range = np.arange(self.year_start, self.year_end + 1)
        data = np.zeros(len(self.year_range))
        self.cal_death_rate_df = pd.DataFrame({'year': self.year_range,
                                             'death_rate': data}, index=self.year_range)
        self.population_df = self.population_df.loc[(self.population_df['year']>= year_start) & (
            self.population_df['year'] <= year_end)]
        self.gdp_df = self.gdp_df.loc[(self.gdp_df['year']>= year_start) & (self.gdp_df['year'] <= year_end)]

    def optim_variable(self, x, bounds):
        self.database = {tuple(x): {}}
        self.current_iter = -1
        self._log_iteration_vector(x)
        x_opt, f, d = fmin_l_bfgs_b(self.eval_all, x, fprime=self.FD_compute_all, bounds=bounds, maxfun=10000, approx_grad=0,
                                    maxiter=1000, m=len(x), iprint=1, pgtol=1.e-9, factr=1., maxls=2 * len(x), callback=self.__log_iteration_vector)
        self.f = f
        self.d = d
        self.x_opt = x_opt
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
        # Initialise everything
        # self.set_data()
        self.compute_death_rate(x)
        # Compute delta
        self.compute_mod_func()
        self.database[tuple(x)] = {'cst_delta_pib': self.mod_func}
        return self.mod_func

    def compute_mod_func(self):
        self.delta_br = self.comp_delta_br()
        #self.delta_var = self.comp_delta_var()elf.cal_death_rate_df
        self.delta_sum = self.delta_br.fillna(0)
        self.delta_br =  self.delta_br.fillna(0)
        func_manager = FunctionManager()
        #-- reference square due to delta**2
        func_manager.configure_smooth_log(False, 1e10)
        func_manager.add_function('cst_delta_pib', np.array(
            self.delta_br), FunctionManager.INEQ_CONSTRAINT, weight=1.)
        func_manager.build_aggregated_functions(eps=1e-3)
        self.mod_func = func_manager.mod_obj
        return self.mod_func

    def FD_compute_all(self, x):
        grad_eval = FDGradient(1, self.eval_all, fd_step=1.e-12)
        grad_eval.set_multi_proc(False)
        outputs_grad = grad_eval.grad_f(x)
        #print(outputs_grad)
        return outputs_grad

    def comp_delta_br(self):
        """ Compute (y_ref - y_comp)^2 and returns a series
        Inputs: 2 dataframes"""
        br_base_df = self.deathrate_df
        delta = (br_base_df['death_rate'] -
                 self.cal_death_rate_df['death_rate']) / (br_base_df['death_rate'])
        
#         delta.iloc[1] = delta.iloc[1]/2
#         delta.iloc[2] =  delta.iloc[2]/2
#         delta.iloc[3] = delta.iloc[3]/2 
#         delta.iloc[4] = delta.iloc[4]/2
#         delta.iloc[0] = delta.iloc[0]/2
#         delta.iloc[-2] = delta.iloc[-4]*2 
#         delta.iloc[-3] = delta.iloc[-5]*2
        delta.iloc[56] = delta.iloc[56]*2
        delta.iloc[58] = delta.iloc[58]*2
        delta.iloc[59] = delta.iloc[59]*2
        delta.iloc[-1] = delta.iloc[-1]
        delta = delta.fillna(0)
        absdelta = np.sign(delta) * delta
        return absdelta

    def comp_delta_var(self):
        """Compute the difference in variation"""
        var_serie = pd.Series(
            np.zeros(len(self.year_range)), index=self.year_range)
        dr_base_df = self.deathrate_df
        calib_dr_df = self.cal_death_rate_df
        for year in self.year_range[1:]:
            dr_base_y =  dr_base_df.loc[year, 'death_rate'] 
            p_dr_base =  dr_base_df.loc[year - 1, 'death_rate'] 
            dr_y = calib_dr_df.loc[year, 'death_rate']
            p_dr = calib_dr_df.loc[year - 1, 'death_rate']
            var_dr_base = dr_base_y - p_dr_base
            var_dr_y = dr_y - p_dr
            delta_var = var_dr_base - var_dr_y
            var_serie.loc[year] = delta_var**2
        return var_serie

    def compute_death_rate(self, x):
        """
        death rate formula from
        McIsaac, F., 2020. A Representation of the World Population Dynamics for Integrated Assessment Models. 
        Environmental Modeling & Assessment, 25(5), pp.611-632.
        inputs: dataframe death rate, df population, df gdp, x: list of parameters
        return: dataframe of estimated death rate with x
        """
        br_upper_a = x[0]
        br_lower_a = x[1]
        delta = x[2]
        phi = x[3]
        nu = x[4]
        self.delta = delta
        self.br_upper_a = br_upper_a
        self.br_lower_a = br_lower_a
        self.phi = phi
        self.nu = nu
        gdp = self.gdp_df['GDP']
        pop = self.population_df['population']
        death_rate = br_upper_a + (br_lower_a - br_upper_a)/(1+np.exp(-delta*(gdp/pop-phi)))**(1/nu)
        self.cal_death_rate_df['death_rate'] = death_rate.fillna(0)
        self.cal_death_rate_df = self.cal_death_rate_df.fillna(0)
        return self.cal_death_rate_df
    
    def compute_estimated_death_rate(self, x):
        br_upper_a = x[0]
        br_lower_a = x[1]
        delta = x[2]
        phi = x[3]
        nu = x[4]
        new_year_range = np.arange(1800, 2300)
        year_below = np.arange(1800, 1960)
        year_above = np.arange(1960, 2300)
        x = year_above - 1960
        #COmpute gdp 
        gdp_data = 2e8*x**3 + 1e10*x**2 + 6e11*x + 2e13
        gdp = pd.Series(gdp_data)
        gdp.index = year_above
        #Remove ngative gdp and replace by a value
        #gdp[gdp<0] = 5e12
        #compute pop 
        data_pop = 3e9+8e7*x
        pop = pd.Series(data_pop)
        pop.index = year_above
        #pop[pop<0] = 5e8
        size = 1960 - 1800 
        low_gdpcapita = np.array(np.linspace(1000, 6666,size))
        low_gdp_capita = list(low_gdpcapita)
        gdp_capita = list(gdp / pop)
        low_gdp_capita.extend(gdp_capita)
        gdp_c = pd.Series(low_gdp_capita)
        gdp_c.index = new_year_range
        death_rate = br_upper_a + (br_lower_a - br_upper_a)/(1+np.exp(-delta*(gdp_c- phi)))**(1/nu)
        x = new_year_range - 1800
        estimated_dr_df = pd.DataFrame({'year': new_year_range, 'death_rate': death_rate})
        estimated_dr_df.index = new_year_range
        self.estimated_dr_df = estimated_dr_df
        
        return estimated_dr_df     
        
    def plot_br_gdp(self):
        """
        Plot death rate vs gdp per age group: historical data and estimation
        """
        age = self.age
        new_chart = TwoAxesInstanciatedChart('gdp per capita', 'death rate')
        gdp_pc = self.gdp_df["GDP"]/self.population_df['population']
        new_chart.add_series(InstanciatedSeries(
           gdp_pc.tolist(), self.deathrate_df['death_rate'].tolist(), f'Death rate vs gdp per capita for age {age}', InstanciatedSeries.SCATTER_DISPLAY))
        new_chart.add_series(InstanciatedSeries(
           gdp_pc.tolist(), self.cal_death_rate_df['death_rate'].tolist(), f'estimated death rate vs gdp per capita for age {age}', InstanciatedSeries.LINES_DISPLAY))

        new_chart.to_plotly().show()
        
    def plot_deathrate_years(self):
        age = self.age 
        base_br = self.deathrate_df['death_rate'][self.deathrate_df['year']<=2020]
        cal_br = self.cal_death_rate_df['death_rate'][self.cal_death_rate_df['year']<=2020]
        new_chart = TwoAxesInstanciatedChart('years', 'death_rate')
        new_chart.add_series(InstanciatedSeries(list(self.cal_death_rate_df['year']), list(cal_br), 
                                                f'estimated death rate per year for age {age}', InstanciatedSeries.LINES_DISPLAY))
        new_chart.add_series(InstanciatedSeries(
           list(self.deathrate_df['year']), list(base_br), f'Birth rate for age {age}', InstanciatedSeries.SCATTER_DISPLAY))
        new_chart.to_plotly().show()
        
    def plot_estimated_dr(self):
        age = self.age
        new_chart = TwoAxesInstanciatedChart('year', 'death rate', chart_name = f'estimated death rate for age {age}')
        new_chart.add_series(InstanciatedSeries(
           list(self.estimated_dr_df['year']), list(self.estimated_dr_df['death_rate']), f'estimated death rate for age {age}', InstanciatedSeries.LINES_DISPLAY))
        
        new_chart.to_plotly().show()        
        
    def plot_comparison_estimated_df(self, dr_df_one, dr_df_sec):      
        age = self.age
        ihme_dr = self.ihme_dr_df[age]
        un_dr = self.un_dr_df[age]
        hist_dr = self.deathrate_df_full[self.deathrate_df_full['age'] == age]
        new_chart = TwoAxesInstanciatedChart('year', 'death rate', chart_name = f'Comparison of estimated death rates for age {age}')
        new_chart.add_series(InstanciatedSeries(
           list(dr_df_one['year']), list(dr_df_one['death_rate']), 'x_opt_one', InstanciatedSeries.LINES_DISPLAY))
        new_chart.add_series(InstanciatedSeries(
           list(dr_df_sec['year']), list(dr_df_sec['death_rate']), 'x_opt_sec', InstanciatedSeries.LINES_DISPLAY))
        new_chart.add_series(InstanciatedSeries(
           list(self.ihme_dr_df['year']), list(ihme_dr), 'ihme death rate', InstanciatedSeries.LINES_DISPLAY))
        new_chart.add_series(InstanciatedSeries(
           list(self.un_dr_df['year']), list(un_dr), 'un death rate', InstanciatedSeries.LINES_DISPLAY))
        new_chart.add_series(InstanciatedSeries(
           list(hist_dr['year']), list(hist_dr['death_rate'][hist_dr['year']<2020]), 'historical un death rate', InstanciatedSeries.LINES_DISPLAY))
            
        new_chart.to_plotly().show()  
 