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


class HasslerPib(BaseCalib):
    """
    Compute the Hassler et al. production function
    Hassler, J., Krusell, P. and Olovsson, C., 2012. Energy-saving technical change (No. w18456).
    National Bureau of Economic Research.
    available at: https://www.nber.org/system/files/working_papers/w18456/w18456.pdf
    """

    def compute_estimated_pib(self, year, x):
        capital_share = x[5]
        elast_KL_E = x[6]
        energy_share = x[7]
#         energy_y = self.energy_df.loc[year, 'Energy']/self.energy_df.loc[self.year_range[0], 'Energy'] #in TWh
#         population_y = self.population_df.loc[year, 'population']/self.population_df.loc[self.year_range[0], 'population'] #In millions of people
#         capital_y = self.capital_df.loc[year, 'capital (trill 2011)']/self.capital_df.loc[self.year_range[0], 'capital (trill 2011)'] #In trill$
#         productivity_y = self.productivity_df.loc[year, 'productivity']/ self.productivity_df.loc[self.year_range[0], 'productivity']
#         energy_intensity_y = self.energy_intens[year]/ self.energy_intens[self.year_range[0]]
        energy_y = self.energy_df.loc[year, 'Energy']  # in TWh
        # In millions of people
        population_y = self.population_df.loc[year, 'population']
        capital_y = self.capital_df.loc[year,
                                        'capital (trill 2011)']  # In trill$
        productivity_y = self.productivity_df.loc[year, 'productivity']
        energy_intensity_y = self.energy_intens[year]
        # Cobb-Douglas part linking Capital and Labour
        cobb_douglas = ((productivity_y * (capital_y**capital_share) * ((population_y / 1000)
                                                                        ** (1 - capital_share)))) ** ((elast_KL_E - 1) / elast_KL_E)
        energy_part = (((energy_intensity_y * energy_y / 10000))
                       ** ((elast_KL_E - 1) / elast_KL_E))
        # 2-level nested CES function, links the Cobb-Douglas and the Energy
        gross_output_rel = ((1 - energy_share) * cobb_douglas + energy_share *
                            energy_part) ** (elast_KL_E / (elast_KL_E - 1))
        #gross_output = gross_output_rel * self.pib_year_start
        gross_output = gross_output_rel
        return gross_output

#     def compute_productivity(self, year):
#         """
#         overload with hassler paper data
#         """
#         productivity = self.productivity_start * \
#             (1 + self.productivity_gr)**(year - self.year_start + 1)
#         return(productivity)

    def eval_all(self, x):
        """
        Base eval all 
        """
        self.energy_factor = x[0]
        self.gr_rate_energy = x[1]
        self.productivity_start = x[2]
        self.productivity_gr = x[3]
        self.decline_rate_tfp = x[4]
        # Initialise everything
        self.set_data()
        year_range = self.year_range
        self.productivity_df.loc[year_range[0],
                                 'productivity_gr'] = self.productivity_gr
        self.productivity_df.loc[year_range[0],
                                 'productivity'] = self.productivity_start
        self.energy_intens[year_range[0]] = self.energy_factor
        # COmpute productivity and energy intensity
        for year in year_range[1:]:
            #             self.productivity_df.loc[year, 'productivity_gr'] = self.compute_productivity_growth_rate(
            #                 year)
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


energy_factor = 0.3  # 0.5
gr_rate_ener = 0.019  # 0.019
productivity_start = 2.1  # 1
productivity_gr = 0.199  # 0.076
decline_rate_tfp = -0.009  # 0.005
capital_share = 0.515  # 0.25
elast_KL_E = 0.48898  # 0.02
energy_share = 0.04617  # 0.09
x = [energy_factor, gr_rate_ener, productivity_start, productivity_gr, decline_rate_tfp, capital_share,
     elast_KL_E, energy_share]
Test = HasslerPib()
Test.eval_all(x)
# print(Test.pib_df)
# bounds = [(0.1, 5), (0.001, 0.99), (0.1, 1.1), (-0.1, 0.2),
#           (0.0001, 0.01), (0.09, 0.8), (0.01, 5), (0.1, 0.99)]
# bounds = [(0.9, 1.1), (0.001, 0.99), (0.01, 2.1), (-0.1, 0.2),
#           (0.001, 0.01), (0.09, 0.8), (0.01, 5), (0.28, 0.3)]
bounds = [(0.1, 5), (0.001, 0.99), (0.01, 2.1), (-0.1, 0.2),
          (-0.01, 0.01), (0.09, 0.8), (0.01, 5), (0.01, 0.95)]
x_opt = Test.optim_pib(x, bounds)
print(x_opt)
Test.plot_pib()
