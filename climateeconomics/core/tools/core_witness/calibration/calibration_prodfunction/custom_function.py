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
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart,\
    InstanciatedSeries
import numpy as np
import pandas as pd

class CustomPibLogistic(BaseCalib):
    """
    A custom version of production function based on linear energy only 
    """

    def __init__(self):
        super().__init__('custom_energy_GIVEANAME')
        self.logger.info("")

    def eval_all(self, x):
        """
        Base eval all 
        """
        self.x_zero_tfp = x[0]
        self.L_tfp = x[1]
        self.k_tfp = x[2]
        self.x_zero_e = x[3]
        self.L_e = x[4]
        self.k_e = x[5]
        
        # Initialise everything
        self.set_data()
        year_range = self.year_range
        data = np.zeros(len(self.year_range))
        self.energy_intens_df = pd.DataFrame({'year': self.year_range, 'energy_intensity': data,
                                             'energy_intens_gr': data}, index=self.year_range)
        # COmpute pib
        for year in year_range:
            self.productivity_df.loc[year, 'productivity'] = self.compute_productivity(year, 
                                                                self.x_zero_tfp, self.L_tfp, self.k_tfp)
            self.energy_intens_df.loc[year, 'energy_intensity'] = self.compute_energy_intensity(year, 
                                                                        self.x_zero_e, self.L_e, self.k_e)
            self.pib_df.loc[year, 'output'] = self.compute_estimated_pib(
                year, x)
        # Compute delta
        self.compute_mod_func()
        self.database[tuple(x)] = {'cst_delta_pib' : self.mod_func}
        return self.mod_func

    def compute_estimated_pib(self, year, x):
        energy_factor = x[6]
        pop_factor = x[7]
        b = x[8]
        c =x[9]
        alpha = x[10]
        beta = x[11]
        gamma = x[12]
        theta = x[13]
#         energy_intensity_y = self.energy_intens_df.loc[year, 'energy_intensity']/ self.energy_intens_df.loc[self.year_range[0], 'energy_intensity']
#         population_y = self.population_df.loc[year, 'population']/ self.population_df.loc[self.year_range[0], 'population']
#         energy_y = self.energy_df.loc[year, 'Energy']/ self.energy_df.loc[self.year_range[0], 'Energy']
        energy_intensity_y = self.energy_intens_df.loc[year, 'energy_intensity']
        productivity_y = self.productivity_df.loc[year, 'productivity']
        capital_y = self.capital_df.loc[year,
                                        'capital (trill 2011)']  
        population_y = self.population_df.loc[year, 'population']
        energy_y = self.energy_df.loc[year, 'Energy']
#         cobb_douglas =  (productivity_y * (capital_y**capital_share)
#                         * ((population_y / 1000) ** (1 - capital_share)))
        #energy_intensity_y = 1*(1+5.06e-02)**(year-self.year_start+1)
        #output = cobb_douglas + a*energy_y*energy_intensity_y + cst
#        output =  b*productivity_y*(c*(population_y*pop_factor)+(1-c)*capital_y)+(1-b)*a*energy_y*energy_intensity_y 
        output = (b*productivity_y*(c*(pop_factor*population_y)**alpha+(1-c)*capital_y)**beta+(1-b)*
                  (energy_factor*energy_y)**gamma*energy_intensity_y)**theta  
        return output 
    
    def compute_energy_intensity(self, year, x_zero_e, L_e, k_e):
        energy_intensity = L_e/(1+ np.exp(-k_e*(year -x_zero_e)))
        return energy_intensity

#     def compute_productivity(self, year, x_zero_tfp, L_tfp, k_tfp):
#         productivity = L_tfp/(1+ np.exp(-k_tfp*(year-x_zero_tfp)))
#         return productivity
    
    def projection_eval(self,x):
        self.x_zero_tfp = x[0]
        self.L_tfp = x[1]
        self.k_tfp = x[2]
        self.x_zero_e = x[3]
        self.L_e = x[4]
        self.k_e = x[5]

        self.set_data(2015, 2050)
        #create and concat df 
        self.create_projection_df()
        ### productivities
        years_tot = np.arange(1965, self.year_end+1)
        data = np.zeros(len(years_tot))
        self.productivity_df = pd.DataFrame({'year': years_tot, 'productivity': data,
                                             'productivity_gr': data}, 
                                            index =years_tot)
        self.energy_intens_df = pd.DataFrame({'year': years_tot, 'energy_intensity': data,
                                                 'energy_intens_gr': data},
                                                 index= years_tot)
        for year in years_tot:
            self.productivity_df.loc[year, 'productivity'] = self.compute_productivity(year,
                                                                                       self.x_zero_tfp, self.L_tfp, self.k_tfp)
            self.energy_intens_df.loc[year, 'energy_intensity'] = self.compute_energy_intensity(year,
                                                                                                 self.x_zero_e, self.L_e, self.k_e)
        #COmpute PIB and capital 
        for year in self.year_range:
            pib_p = self.pib_df.loc[year-1, 'output']
            capital_p = self.capital_df.loc[year-1, 'capital (trill 2011)']
            capital = 0.9 * capital_p + 0.2*pib_p
            self.capital_df.loc[year,'capital (trill 2011)'] = capital
            self.pib_df.loc[year, 'output']  = self.compute_estimated_pib(year, x)
        return self.pib_df 


class CustomPib(BaseCalib):
    """
    A custom version of production function based on linear energy only 
    """

    def __init__(self):
        super().__init__('new_custom_energy_new_pib_lalala')
        self.logger.info("")

    def eval_all(self, x):
        """
        Base eval all 
        """
        self.productivity_gr_start = x[0]
        self.productivity_start = x[1]
        self.decline_rate_tfp = x[2]
        self.init_energy_prod = x[3]
        self.init_gr_energy = x[4]
        self.decline_rate_energy = x[5] 
        # Initialise everything
        self.set_data()
        year_range = self.year_range
        data = np.zeros(len(self.year_range))
        self.productivity_df.loc[year_range[0],
                                 'productivity_gr'] = self.productivity_gr_start
        self.productivity_df.loc[year_range[0],
                                 'productivity'] = self.productivity_start
        self.energy_intens_df = pd.DataFrame({'year': self.year_range, 'energy_intensity': data,
                                             'energy_intens_gr': data}, index=self.year_range)
        self.energy_intens_df.loc[year_range[0], 'energy_intensity'] = self.init_energy_prod
        self.energy_intens_df.loc[year_range[0],'energy_intens_gr'] = self.init_gr_energy                         
        # COmpute productivity and energy intensity
        for year in year_range[1:]:
            self.productivity_df.loc[year, 'productivity_gr'] = self.compute_productivity_growth_rate(
                year)
            self.productivity_df.loc[year, 'productivity'] = self.compute_productivity(
                year)
            self.energy_intens_df.loc[year, 'energy_intens_gr'] = self.compute_energy_intens_gr(year)
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
        b = x[6]
        c =x[7]
#         energy_factor = x[8]
#         pop_factor = x[9]
#         alpha = x[10]
#         beta = x[11]
#         gamma = x[12]
#         theta = x[13]
        alpha = x[8]
        beta = x[9]
        gamma = x[10]
        theta = x[11]
        energy_factor= 1e-4
        pop_factor = 1e-3
#         energy_intensity_y = self.energy_intens_df.loc[year, 'energy_intensity']/ self.energy_intens_df.loc[self.year_range[0], 'energy_intensity']
#         population_y = self.population_df.loc[year, 'population']/ self.population_df.loc[self.year_range[0], 'population']
#         energy_y = self.energy_df.loc[year, 'Energy']/ self.energy_df.loc[self.year_range[0], 'Energy']
        energy_intensity_y = self.energy_intens_df.loc[year, 'energy_intensity']
        productivity_y = self.productivity_df.loc[year, 'productivity']
        capital_y = self.capital_df.loc[year,
                                        'capital']  
        population_y = self.population_df.loc[year, 'population']
        energy_y = self.energy_df.loc[year, 'Energy']
#         cobb_douglas =  (productivity_y * (capital_y**capital_share)
#                         * ((population_y / 1000) ** (1 - capital_share)))
        #energy_intensity_y = 1*(1+5.06e-02)**(year-self.year_start+1)
        #output = cobb_douglas + a*energy_y*energy_intensity_y + cst
#        output =  b*productivity_y*(c*(population_y*pop_factor)+(1-c)*capital_y)+(1-b)*a*energy_y*energy_intensity_y 
        economic_part = productivity_y*(c*(pop_factor*population_y)**alpha+(1-c)*(capital_y)**beta)
        energy_part = energy_intensity_y*(energy_factor*energy_y)**gamma
        output = (b*economic_part + (1-b)*energy_part)**theta    
#         output = (b*productivity_y*(c*(pop_factor*population_y)**alpha+(1-c)*(capital_y)**beta)+(1-b)*
#                   ((energy_factor*energy_y)**gamma*energy_intensity_y)**theta)  
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
    
    def projection_eval(self,x):
        self.productivity_gr_start = x[0]
        self.productivity_start = x[1]
        self.decline_rate_tfp = x[2]
        self.init_energy_prod = x[3]
        self.init_gr_energy = x[4]
        self.decline_rate_energy = x[5] 
        
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
        self.productivity_df.loc[1980,
                                 'productivity_gr'] = self.productivity_gr_start
        self.productivity_df.loc[1980,
                                 'productivity'] = self.productivity_start
        self.energy_intens_df.loc[1980, 'energy_intensity'] = self.init_energy_prod
        self.energy_intens_df.loc[1980,'energy_intens_gr'] = self.init_gr_energy
        self.year_start = 1980            
        for year in years_tot[1:]:
            self.productivity_df.loc[year, 'productivity_gr'] = self.compute_productivity_growth_rate(
                year)
            self.productivity_df.loc[year, 'productivity'] = self.compute_productivity(
                year)
            self.energy_intens_df.loc[year, 'energy_intens_gr'] = self.compute_energy_intens_gr(year)
            self.energy_intens_df.loc[year, 'energy_intensity'] = self.compute_energy_intensity(year)
        print('2020 energy_gr', self.energy_intens_df.loc[2020, 'energy_intens_gr'])
        print('2020 intensity', self.energy_intens_df.loc[2020, 'energy_intensity'])  
        print('2020 productivity', self.productivity_df.loc[2020, 'productivity'])  
        print('2020 productivity gr', self.productivity_df.loc[2020, 'productivity_gr'])                                                                                    
        #COmpute PIB and capital 
        self.year_start = 2020
        for year in self.year_range:
            pib_p = self.pib_df.loc[year-1, 'output']
            capital_p = self.capital_df.loc[year-1, 'capital']
            capital = 0.92 * capital_p + 0.25*pib_p
            self.capital_df.loc[year,'capital'] = capital
            self.pib_df.loc[year, 'output']  = self.compute_estimated_pib(year, x)
        return self.pib_df 

class CustomPibCD(BaseCalib):
    """
    A custom version of production function based on linear energy only 
    """

    def __init__(self):
        super().__init__('new_custom_energy_cd_new_pib_NAMENAMENAME')
        self.logger.info("")

    def eval_all(self, x):
        """
        Base eval all 
        """
        self.productivity_gr_start = x[4]
        self.productivity_start = x[5]
        self.decline_rate_tfp = x[6]
        self.init_energy_prod = x[0]
        self.init_gr_energy = x[1]
        self.decline_rate_energy = x[2] 

        # Initialise everything
        self.set_data()
        year_range = self.year_range
        data = np.zeros(len(self.year_range))
        self.productivity_df.loc[year_range[0],
                                 'productivity_gr'] = self.productivity_gr_start
        self.productivity_df.loc[year_range[0],
                                 'productivity'] = self.productivity_start
        self.energy_intens_df = pd.DataFrame({'year': self.year_range, 'energy_intensity': data,
                                             'energy_intens_gr': data}, index=self.year_range)
        self.energy_intens_df.loc[year_range[0], 'energy_intensity'] = self.init_energy_prod
        self.energy_intens_df.loc[year_range[0],'energy_intens_gr'] = self.init_gr_energy                         
        # COmpute productivity and energy intensity
        for year in year_range[1:]:
            self.productivity_df.loc[year, 'productivity_gr'] = self.compute_productivity_growth_rate(
                year)
            self.productivity_df.loc[year, 'productivity'] = self.compute_productivity(
                year)
            self.energy_intens_df.loc[year, 'energy_intens_gr'] = self.compute_energy_intens_gr(year)
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
        energy_k = x[3]
        b = x[4]
        alpha = x[5]
#         energy_intensity_y = self.energy_intens_df.loc[year, 'energy_intensity']/ self.energy_intens_df.loc[self.year_range[0], 'energy_intensity']
#         population_y = self.population_df.loc[year, 'population']/ self.population_df.loc[self.year_range[0], 'population']
#         energy_y = self.energy_df.loc[year, 'Energy']/ self.energy_df.loc[self.year_range[0], 'Energy']
        energy_intensity_y = self.energy_intens_df.loc[year, 'energy_intensity']
        productivity_y = self.productivity_df.loc[year, 'productivity']
        capital_y = self.capital_df.loc[year,
                                        'capital']  
        population_y = self.population_df.loc[year, 'population']
        energy_y = self.energy_df.loc[year, 'Energy']
#         cobb_douglas =  (productivity_y * (capital_y**capital_share)
#                         * ((population_y / 1000) ** (1 - capital_share)))
        #energy_intensity_y = 1*(1+5.06e-02)**(year-self.year_start+1)
        #output = cobb_douglas + a*energy_y*energy_intensity_y + cst
#        output =  b*productivity_y*(c*(population_y*pop_factor)+(1-c)*capital_y)+(1-b)*a*energy_y*energy_intensity_y 
#         economic_part = productivity_y*(c*(pop_factor*population_y)**alpha+(1-c)*(capital_y)**beta)
#         energy_part = energy_intensity_y*(energy_factor*energy_y)**gamma
# #         output = (b*economic_part + (1-b)*energy_part)**theta    
#         output = (b*productivity_y*(c*(pop_factor*population_y)**alpha+(1-c)*(capital_y)**beta)+(1-b)*
#                   ((energy_factor*energy_y)**gamma*energy_intensity_y)**theta)
        output = (1-b)*energy_k*energy_y*energy_intensity_y + b*productivity_y*capital_y**alpha*(population_y/1000)**(1-alpha) 
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
    
    def projection_eval(self,x):
        self.productivity_gr_start = x[0]
        self.productivity_start = x[1]
        self.decline_rate_tfp = x[2]
        self.init_energy_prod = x[3]
        self.init_gr_energy = x[4]
        self.decline_rate_energy = x[5] 
        
        self.set_data(2015, 2050)
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
        self.productivity_df.loc[self.year_range[0],
                                 'productivity_gr'] = self.productivity_gr_start
        self.productivity_df.loc[self.year_range[0],
                                 'productivity'] = self.productivity_start
        self.energy_intens_df.loc[self.year_range[0], 'energy_intensity'] = self.init_energy_prod
        self.energy_intens_df.loc[self.year_range[0],'energy_intens_gr'] = self.init_gr_energy
                                 
        for year in self.year_range[1:]:
            self.productivity_df.loc[year, 'productivity_gr'] = self.compute_productivity_growth_rate(
                year)
            self.energy_intens_df.loc[year, 'energy_intens_gr'] = self.compute_energy_intens_gr(year)
            self.productivity_df.loc[year, 'productivity'] = self.compute_productivity(year)
            self.energy_intens_df.loc[year, 'energy_intensity'] = self.compute_energy_intensity(year)
                                                                                                 
        #COmpute PIB and capital 
        for year in self.year_range:
            pib_p = self.pib_df.loc[year-1, 'output']
            capital_p = self.capital_df.loc[year-1, 'capital (trill 2011)']
            capital = 0.9 * capital_p + 0.2*pib_p
            self.capital_df.loc[year,'capital (trill 2011)'] = capital
            self.pib_df.loc[year, 'output']  = self.compute_estimated_pib(year, x)
        return self.pib_df 

Test = CustomPib()
x_zero_tfp = 2020
L_tfp = 5
k_tfp = 1
x_zero_e = 2020
L_e = 5
k_e = 1
energy_factor = 3.75e-05
pop_factor = 1e-3
b = 0.1
c =0.3
alpha = 0.1
beta = 0.1
gamma = 0.1
theta = 0.1
productivity_gr_start = 0.05
productivity_start =0.5
decline_rate_tfp = 0.005
init_energy_prod = 0.2
init_gr_energy  = 0.1
decline_rate_energy = 0.005 
x = [productivity_gr_start, productivity_start, decline_rate_tfp, init_energy_prod, 
     init_gr_energy, decline_rate_energy,  b, c, alpha, beta, gamma, theta]
bounds = [(0.001, 0.2),(0.01, 2),(0.00001, 0.1), (0.01, 4), (0.01, 0.2), (0.00001, 0.1),
    (0.01, 0.4), (0.1, 0.9), (0.01, 5.), (0.01, 5.), (0.01, 5.), (0.01, 5.)]
#x = [x_zero_tfp, L_tfp, k_tfp, x_zero_e, L_e, k_e, b, c, pop_factor, energy_factor, alpha, beta, gamma, theta]
#x = [productivity_gr_start, productivity_start, decline_rate_tfp, capital_share, a, cst]
#x_dice = [5.15, 0.076, 0.005, 0.3]
# bounds = [(1980., 2050.), (1., 15.), (-1, 1), (1980., 2050.), (1., 15.), (-1, 1), 
#           (0.1, 0.9), (0.1, 0.9), (1.e-5, 1.e7), (1.e-5, 1.e7), (-5., 5.), (-5., 5.), (-5., 5.), (-10., 5.)]
# # #bounds = [(-0.1, 0.2), (0.1, 1.1), (0.0001, 0.01), (0.09, 0.8), (1e-8, 1e10), (-1e2,1e2)]
#HERE FOR THE OPTIM 
# x_opt = Test.optim_pib(x, bounds)
# Test.logger.info(Test.pib_df)
# print(x_opt)
# print(Test.pib_df)
# Test.plot_pib()
#print(Test.energy_intens_df)

#Calibration custom CD 
# Test = CustomPibCD()
# energy_factor = 3.99261872e+00
# init_gr_energy = 8.22538141e-02
# decline_rate_energy = 4.02875361e-02
# energy_k = 3.00000000e-05
# alpha = 0.3
# x = [energy_factor, init_gr_energy, decline_rate_energy, energy_k, productivity_gr_start,
#      productivity_start, decline_rate_tfp, b, alpha]
# bounds = [(3., 4.5), (0.07, 0.1), (4.e-2, 5.e-2), (2e-5, 4e-5), (0.001, 0.2),(0.01, 1),(0.00001, 0.1), (0.01, 0.4), (0.2, 0.8)]
# x_opt = Test.optim_pib(x, bounds)
# Test.logger.info(Test.pib_df)
# print(x_opt)
# print(Test.pib_df)
# Test.plot_pib()
#TEST EVAL CUSTOM PIB
# x_opt = 
# Test.projection_eval(x_opt)
# Test.plot_pib2()

#TEST EVAL projection with optimal x
# x_opt = [0.13769917, 0.67142035, 0.02176604, 3.99958053, 0.02074138, 0.01282409, 0.01,
#             0.25343445, 0.12852113, 0.44925737, 0.52398495, 1.39834205]
x_opt2 = [0.10994158, 0.54487279, 0.02351234, 2.14301717, 0.01123201, 0.01345699,0.01, 0.29098974, 0.10810935, 0.1924566,  0.24858645, 2.75040992]
Test.eval_all(x_opt2)
Test.projection_eval(x_opt2)
Test.plot_pib()
# print(Test.energy_intens_df.loc[Test.energy_intens_df['year'] == 2015])
# print(Test.productivity_df.loc[Test.productivity_df['year'] == 2015])
