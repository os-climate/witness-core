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
import pandas as pd

class ZhaZhouSimplePib(BaseCalib):
    """
    Compute the Zha, Zhou production function. 
    Zha, D. and Zhou, D., 2014. The elasticity of substitution and the way of 
    nesting CES production function with emphasis on energy input.
    Applied energy, 130, pp.793-798.
    WITH MODIFICATIONS: added energy intensity 
    And productivity evolution
    """
    def __init__(self):
        super().__init__('ZhaZhousinplified_v1')
    
    def eval_all(self, x):
        """Base eval all 
        """
        self.energy_factor = x[0]
        self.decline_rate_energy = x[1]
        self.productivity_start = x[2]
        self.productivity_gr_start = x[3]
        self.decline_rate_tfp = x[4]
        self.init_gr_energy = x[9]
        # Initialise everything
        #self.set_data()
        year_range = self.year_range
        data = np.zeros(len(self.year_range))
        self.energy_intens_df = pd.DataFrame({'year': self.year_range, 'energy_intensity': data,
                                             'energy_intens_gr': data}, index=self.year_range)
        self.productivity_df.loc[year_range[0],
                                 'productivity_gr'] = self.productivity_gr_start
        self.productivity_df.loc[year_range[0],
                                 'productivity'] = self.productivity_start
        self.energy_intens_df.loc[year_range[0], 'energy_intensity'] = self.energy_factor
        self.energy_intens_df.loc[year_range[0],'energy_intens_gr'] = self.init_gr_energy
        # COmpute productivity and energy intensity
        for year in year_range[1:]:
            self.productivity_df.loc[year, 'productivity_gr'] = self.compute_productivity_growth_rate(
                year)
            self.productivity_df.loc[year, 'productivity'] = self.compute_productivity(
                year)
            self.energy_intens_df.loc[year,'energy_intens_gr']= self.compute_energy_intens_gr(year)
            self.energy_intens_df.loc[year, 'energy_intensity'] = self.compute_energy_intensity(year)
        # COmpute pib
        for year in year_range:
            self.pib_df.loc[year, 'output'] = self.compute_estimated_pib(
                year, x)
        # Compute delta
        self.compute_mod_func()
        self.database[tuple(x)] = {'cst_delta_pib' : self.mod_func}
        return self.mod_func

        
    def compute_estimated_pib(self, year, x):
        alpha = x[5]
        gamma = x[6]
        small_a = x[7]
        b = x[8]
        k = x[10]
        energy_y = self.energy_df.loc[year, 'Energy']  #in TWh
        population_y = self.population_df.loc[year, 'population']#In millions of people
        capital_y = self.capital_df.loc[year, 'capital (trill 2011)'] #In trill$
        productivity_y = self.productivity_df.loc[year, 'productivity']
        energy_intensity_y = self.energy_intens_df.loc[year, 'energy_intensity']
#         energy_y = self.energy_df.loc[year, 'Energy']/self.energy_df.loc[self.year_range[0], 'Energy'] #in TWh
#         population_y = self.population_df.loc[year, 'population']/self.population_df.loc[self.year_range[0], 'population'] #In millions of people
#         capital_y = self.capital_df.loc[year, 'capital (trill 2011)']/self.capital_df.loc[self.year_range[0], 'capital (trill 2011)'] #In trill$
#         productivity_y = self.productivity_df.loc[year, 'productivity']/ self.productivity_df.loc[self.year_range[0], 'productivity']
#         energy_intensity_y = self.energy_intens[year]/ self.energy_intens[self.year_range[0]]
        # Function
        cobb_douglas = productivity_y*(small_a * (capital_y) **(-alpha) + (1- small_a) * (population_y)**(- alpha))
        output_rel = b*cobb_douglas**(-gamma/alpha) + (1-b)* (energy_intensity_y*k*energy_y) **gamma
        #output = output_rel * self.pib_year_start #Value at t=0 of pib  
        output = output_rel
        return output 
    
    def compute_energy_intens_gr(self, year):
        t = ((year - self.year_start) / 1) + 1
        energy_intens_gr = self.init_gr_energy * \
        np.exp(-self.decline_rate_energy * (t - 1))
        return energy_intens_gr
        
    def compute_energy_intensity(self, year):
        p_energy_intens = self.energy_intens_df.loc[year - 1, 'energy_intensity']
        p_energy_intens_gr = self.energy_intens_df.loc[year -1, 'energy_intens_gr']
        energy_intens = (p_energy_intens / (1 - (p_energy_intens_gr)))
        return energy_intens


energy_factor = 0.5
gr_rate_ener = 0.019 #0.001
productivity_start = 0.6
productivity_gr_start = 0.076
decline_rate_tfp = 0.005
alpha = 1.
beta = 1.
gamma = 1.
small_a = 0.5
b =   0.5
decline_rate_energy = 0.001
init_gr_energy = 0.019
k = 1e-3
x = [energy_factor, decline_rate_energy, productivity_start, productivity_gr_start, decline_rate_tfp, 
     alpha, gamma, small_a, b, init_gr_energy, k]
# x = [ 0.33511633,  0.00697586,  0.79924202,  0.16795982,  0.01, -0.68852919,  -1. , 0.39064669 , 0.9, 0.94596054,  0.0698508, 1e-3]
Test = ZhaZhouSimplePib()
#Test.eval_all(x)
bounds = [(0.1, 5), (0.00001,0.01), (0.1, 1.1), (-0.1, 0.2), (0.00001, 0.01), (-1.,5), 
          (0.01, 5.),(0.2 ,0.9), (0.3, 0.95), (0.001, 0.99), (1.e-5, 1.e5)]
#bounds = [(0.1, 5), (0.001, 0.99),(0.1, 1.1), (-0.1, 0.2), (0.0001, 0.01), (-1.,5.), (-1, 5.), (0.01, 5.),(0.2 ,0.9), (0.3, 0.95)]
x_opt = Test.optim_pib(x, bounds)
#x_great = [5.0e-01, 1.90e-02, 6.0e-01, 2.0e-01, 1.0e-04, 2.126, 2.78045, 2.0415, 9.0e-01, 8.638e-01]
Test.eval_all(x_opt)
print(Test.pib_df)
print(Test.productivity_df)
print(Test.energy_intens_df)
Test.plot_pib()