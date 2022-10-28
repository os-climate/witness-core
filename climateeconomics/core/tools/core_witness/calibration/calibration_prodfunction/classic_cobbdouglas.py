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

from climateeconomics.core.core_witness.calibration.calibration_prodfunction.base_calib import BaseCalib
import numpy as np


class CobbDouglasPib(BaseCalib):
    """
    Compute the Hassler et al. production function
    Hassler, J., Krusell, P. and Olovsson, C., 2012. Energy-saving technical change (No. w18456).
    National Bureau of Economic Research.
    available at: https://www.nber.org/system/files/working_papers/w18456/w18456.pdf
    """

    def __init__(self):
        super().__init__('CobbDouglasPib_giveaname')
        self.logger.info("Productivity Base, format: Y = AK^a(L/1000)^(1-a)")

    def eval_all(self, x):
        """
        Base eval all 
        """
        self.productivity_start = x[0]
        self.productivity_gr_start = x[1]
        self.decline_rate_tfp = x[2]
        # Initialise everything
        self.set_data()
        year_range = self.year_range
        self.productivity_df.loc[year_range[0],
                                 'productivity_gr'] = self.productivity_gr_start
        self.productivity_df.loc[year_range[0],
                                 'productivity'] = self.productivity_start
        # COmpute productivity and energy intensity
        for year in year_range[1:]:
            self.productivity_df.loc[year, 'productivity_gr'] = self.compute_productivity_growth_rate(
                year)
            self.productivity_df.loc[year, 'productivity'] = self.compute_productivity(
                year)
        # COmpute pib
        for year in year_range:
            self.pib_df.loc[year, 'output'] = self.compute_estimated_pib(
                year, x)
        # Compute delta
        self.compute_mod_func()
        self.database[tuple(x)] = {'cst_delta_pib' : self.mod_func}
        return self.mod_func

    def compute_estimated_pib(self, year, x):
        capital_share = x[3]
#         energy_y = self.energy_df.loc[year, 'Energy']/self.energy_df.loc[self.year_range[0], 'Energy'] #in TWh
#         population_y = self.population_df.loc[year, 'population']/self.population_df.loc[self.year_range[0], 'population'] #In millions of people
#         capital_y = self.capital_df.loc[year, 'capital (trill 2011)']/self.capital_df.loc[self.year_range[0], 'capital (trill 2011)'] #In trill$
#         productivity_y = self.productivity_df.loc[year, 'productivity']/ self.productivity_df.loc[self.year_range[0], 'productivity']
        # In millions of people
        population_y = self.population_df.loc[year, 'population']
        capital_y = self.capital_df.loc[year,
                                        'capital (trill 2011)']  # In trill$
        productivity_y = self.productivity_df.loc[year, 'productivity']
        # Cobb-Douglas part linking Capital and Labour
        cobb_douglas = (productivity_y * (capital_y**capital_share)
                        * ((population_y / 1000) ** (1 - capital_share)))
        #gross_output = gross_output_rel * self.pib_year_start
        return cobb_douglas


productivity_start = 1
productivity_gr_start = 0.076
decline_rate_tfp = 0.005
capital_share = 0.3
x = [productivity_start, productivity_gr_start, decline_rate_tfp, capital_share]
Test = CobbDouglasPib()
# x_dice = [5.15, 0.076, 0.005, 0.3]
# Test.eval_all(x_dice)
bounds = [(0.1, 1.1), (-0.1, 0.2), (0.0001, 0.01), (0.09, 0.8)]
x_opt = Test.optim_pib(x, bounds)
Test.logger.info(Test.pib_df)
print(x_opt)
print(Test.pib_df)
Test.plot_pib()
# Test.plot_pib_var()
