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
    Class for calibration of birth rate 
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
        Load required data (csv)
        """
        birthrate_df = pd.read_csv(os.path.join(
            self.base_path, 'data', 'birth_rate_1549-yearly.csv'))
        self.birthrate_df = birthrate_df.set_index(birthrate_df['year'])
        population_df = pd.read_csv(os.path.join(
            self.base_path, 'data', 'population_1960.csv'))
        population_df = population_df.set_index(population_df['year'])
        self.population_df = population_df
        gdp_df = pd.read_csv(os.path.join(
            self.base_path, 'data', 'gdp_ppp_for_calib.csv'))
        self.gdp_df = gdp_df.set_index(gdp_df['year'])
    
    def set_data(self, year_start = 1960, year_end = 2023):
        """
        Set dataframe with right age range (year_start, year_end)
        """
        self.year_start = year_start
        self.year_end = year_end
        self.year_range = np.arange(self.year_start, self.year_end + 1)
        data = np.zeros(len(self.year_range))
        self.cal_birth_rate_df = pd.DataFrame({'year': self.year_range,
                                             'birth_rate': data}, index=self.year_range)
        f2 = interp1d(self.birthrate_df['year'], self.birthrate_df['birth_rate'])
        data_br = f2(self.year_range)
        self.interp_base_birthrate_df = pd.DataFrame({'year': self.year_range,
                                             'birth_rate': data_br},index = self.year_range)
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
        # Compute birth rate
#         print('x',x)
        self.compute_birth_rate(x)
#         print(self.cal_birth_rate_df)
        # Compute delta
        self.compute_mod_func()
        self.database[tuple(x)] = {'cst_delta_pib': self.mod_func}
        return self.mod_func

    def compute_mod_func(self):
        self.delta_br = self.comp_delta_br()
        self.delta_var = self.comp_delta_var()
        self.delta_sum = self.delta_br
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
        return outputs_grad

    def comp_delta_br(self):
        """ Compute (y_ref - y_comp)^2 and returns a series
        Inputs: 2 dataframes"""
        br_base_df = self.interp_base_birthrate_df
        delta = (br_base_df['birth_rate'] -
                 self.cal_birth_rate_df['birth_rate']) / (br_base_df['birth_rate'])
#         delta.iloc[-1] = delta.iloc[-1]*4
#         delta.iloc[-2] =  delta.iloc[-2] *4
#         delta.iloc[-3] = delta.iloc[-3] *3
        absdelta = np.sign(delta) * delta
        return absdelta

    def comp_delta_var(self):
        """Compute the difference in variation"""
        var_serie = pd.Series(
            np.zeros(len(self.year_range)), index=self.year_range)
        br_base_df = self.interp_base_birthrate_df
        calib_br_df = self.cal_birth_rate_df
        for year in self.year_range[1:]:
            br_base_y =  br_base_df.loc[year, 'birth_rate'] 
            p_br_base =  br_base_df.loc[year - 1, 'birth_rate'] 
            br_y = calib_br_df.loc[year, 'birth_rate']
            p_br = calib_br_df.loc[year - 1, 'birth_rate']
            var_br_base = br_base_y - p_br_base
            var_br_y = br_y - p_br
            delta_var = var_br_base - var_br_y
            var_serie.loc[year] = delta_var**2
        return var_serie

    def compute_birth_rate(self, x):
        """Birth rate 
        inputs: dataframe birth rate, df population, df gdp 
        output: df of computed birth rate   
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
        birth_rate = br_upper_a + (br_lower_a - br_upper_a)/(1+np.exp(-delta*(gdp/pop-phi)))**(1/nu)
        self.cal_birth_rate_df['birth_rate'] = birth_rate
        return self.cal_birth_rate_df
            
    def plot_br_gdp(self):
        """
        Plot birth rate vs gdp for historical data and estimated birth rate for comparison
        """
        new_chart = TwoAxesInstanciatedChart('gdp per capita', 'birth rate')
        gdp_pc = self.gdp_df["GDP"]/self.population_df['population']
        new_chart.add_series(InstanciatedSeries(
           list(gdp_pc), list(self.interp_base_birthrate_df['birth_rate']), 'Birth rate 15-49 vs gdp per capita', InstanciatedSeries.SCATTER_DISPLAY))
        new_chart.add_series(InstanciatedSeries(
           list(gdp_pc), list(self.cal_birth_rate_df['birth_rate']), 'estimated Birth rate 15-49 vs gdp per capita', InstanciatedSeries.LINES_DISPLAY))

        new_chart.to_plotly().show()
 