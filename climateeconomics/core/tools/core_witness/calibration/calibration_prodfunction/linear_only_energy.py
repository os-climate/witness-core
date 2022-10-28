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

class EnergyOnlyPib(BaseCalib):
    """
    A test with energy only 
    """

    def __init__(self):
        super().__init__('Linear_only_energy_gdpppp')
        self.logger.info("")

    def eval_all(self, x):
        """
        Base eval all 
        """
#         self.productivity_gr_start = x[0]
#         self.productivity_start = x[1]
#         self.decline_rate_tfp = x[2]
        self.energy_factor = x[0]
        self.init_gr_energy = x[1]
        self.decline_rate_energy = x[2]
        # Initialise everything
        self.set_data()
        year_range = self.year_range
        data = np.zeros(len(self.year_range))
#         self.productivity_df.loc[year_range[0],
#                                  'productivity_gr'] = self.productivity_gr_start
#         self.productivity_df.loc[year_range[0],
#                                  'productivity'] = self.productivity_start
#         # COmpute productivity and energy intensity
#         for year in year_range[1:]:
#             self.productivity_df.loc[year, 'productivity_gr'] = self.compute_productivity_growth_rate(
#                 year)
#             self.productivity_df.loc[year, 'productivity'] = self.compute_productivity(
#                 year)
        self.energy_intens_df = pd.DataFrame({'year': self.year_range, 'energy_intensity': data,
                                             'energy_intens_gr': data}, index=self.year_range)
        self.energy_intens_df.loc[year_range[0], 'energy_intensity'] = self.energy_factor
        self.energy_intens_df.loc[year_range[0],'energy_intens_gr'] = self.init_gr_energy
        #COmpute productivity and energy intensity
        for year in year_range[1:]:
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
#         a = x[4]
#         capital_share = x[3]
#         cst = x[5]
        a = x[3]
        cst = x[4]
#         energy_intensity_y = self.energy_intens_df.loc[year, 'energy_intensity']/ self.energy_intens_df.loc[self.year_range[0], 'energy_intensity']
#         population_y = self.population_df.loc[year, 'population']/ self.population_df.loc[self.year_range[0], 'population']
#         energy_y = self.energy_df.loc[year, 'Energy']/ self.energy_df.loc[self.year_range[0], 'Energy']
        energy_intensity_y = self.energy_intens_df.loc[year, 'energy_intensity']
#         productivity_y = self.productivity_df.loc[year, 'productivity']
#         capital_y = self.capital_df.loc[year,
#                                         'capital (trill 2011)']  
#         population_y = self.population_df.loc[year, 'population']
        energy_y = self.energy_df.loc[year, 'Energy']
#         cobb_douglas =  (productivity_y * (capital_y**capital_share)
#                         * ((population_y / 1000) ** (1 - capital_share)))
        #energy_intensity_y = 1*(1+5.06e-02)**(year-self.year_start+1)
        #output = cobb_douglas + a*energy_y*energy_intensity_y + cst
        output =  a*energy_y*energy_intensity_y + cst
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

    
    def compute_ratio(self):
        pib = self.pib_base_df['GDP'].loc[self.pib_base_df['year']>=1965]
        ratio = pib/ self.energy_df['Energy']
        print(ratio)
        
        new_chart = TwoAxesInstanciatedChart('years', '$/twh')

        new_chart.add_series(InstanciatedSeries(
            self.energy_df['Year'].tolist(), ratio.tolist(), 'pib/energy', InstanciatedSeries.LINES_DISPLAY))

        new_chart.to_plotly().show()

Test = EnergyOnlyPib()
productivity_gr_start = 0.076
productivity_start = 1
decline_rate_tfp = 0.005
capital_share = 0.3
constant = 10
a = 3.75e-05
energy_factor = 1
init_gr_energy = 0.08
decline_rate_energy = 0.005
cst = 1
x = [energy_factor, init_gr_energy, decline_rate_energy, a, constant]
#x = [productivity_gr_start, productivity_start, decline_rate_tfp, capital_share, a, cst]
#x_dice = [5.15, 0.076, 0.005, 0.3]
bounds = [(0.01, 4), (0.01, 0.2), (0.00001, 0.1) , (3e-5,4e-5 ),(1e-3, 1e4)]
#bounds = [(-0.1, 0.2), (0.1, 1.1), (0.0001, 0.01), (0.09, 0.8), (1e-8, 1e10), (-1e2,1e2)]
x_opt = Test.optim_pib(x, bounds)
Test.logger.info(Test.pib_df)
print(x_opt)
print(Test.pib_df)
Test.plot_pib()
#print(Test.energy_intens_df)
Test.plot_pib_energy()