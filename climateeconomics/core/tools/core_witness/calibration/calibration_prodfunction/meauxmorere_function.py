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
from climateeconomics.core.core_witness.calibration.calibration_prodfunction.base_calib_2022 import BaseCalibBis
import numpy as np
import pandas as pd

class MeauxMorerePib(BaseCalibBis):
    """
    A custom version of production function based on 
    """

    def __init__(self):
        super().__init__('meauxmorere_GIVEANAME')
        self.logger.info("")

    def eval_all(self, x):
        """
        Base eval all 
        """
        self.productivity_gr_start = x[0]
        self.productivity_start = x[1]
        self.decline_rate_tfp = x[2]
        self.alpha = x[3]
        #self.gamma = x[4]       
        # Initialise everything
        self.set_data()
        year_range = self.year_range
        data = np.zeros(len(self.year_range))        
        working_pop = self.population_df
        capital_u = self.compute_usable_capital(self.capital_df, self.energy_df)       
        self.productivity_df.loc[year_range[0],
                                 'productivity_gr'] = self.productivity_gr_start
        self.productivity_df.loc[year_range[0],
                                 'productivity'] = self.productivity_start
        for year in year_range[1:]:
            self.productivity_df.loc[year, 'productivity_gr'] = self.compute_productivity_growth_rate(
                year, self.year_start)
            self.productivity_df.loc[year, 'productivity'] = self.compute_productivity(year)
        for year in year_range:
            self.pib_df.loc[year, 'output'] = self.compute_estimated_pib(x, year,self.productivity_df, working_pop, capital_u)
        # Compute delta
        self.compute_mod_func()
        self.database[tuple(x)] = {'cst_delta_pib' : self.mod_func}
        return self.mod_func

    def compute_estimated_pib(self, x, year, productivity_df, working_pop, capital_u):
        """ Compute the gdp 
        inputs: usable capital by year in trill $ , working population by year in million of people,
             productivity by year (no unit), alpha (between 0 and 1) 
        output: gdp in trillion dollars
        """
        alpha = x[3]
        gamma = 1/2
        productivity = productivity_df.loc[year,'productivity']
        working_pop = working_pop.loc[year, 'workforce']
        capital_u = capital_u[year]
        output = productivity * (alpha * capital_u**gamma + (1-alpha)* (working_pop)**gamma)**(1/gamma)      
        return output
     
      
productivity_gr_start = 0.05
productivity_start =0.5
decline_rate_tfp = 0.005
alpha = 0.9
gamma = 1/2
x = [productivity_gr_start, productivity_start, decline_rate_tfp, alpha]
bounds = [(0.001, 0.2),(0.01, 2),(0.00001, 0.1), (0.5, 0.99)]
Test = MeauxMorerePib()
Test.eval_all(x)
# Test.plot_cap_u_pib()
# x_opt = Test.optim_pib(x, bounds)
# Test.eval_all(x_opt)
# print(x_opt)
# Test.plot_pib()    


