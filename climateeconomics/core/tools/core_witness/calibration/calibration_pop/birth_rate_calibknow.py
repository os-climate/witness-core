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

from climateeconomics.core.core_witness.calibration.calibration_pop.base_calib_birthrate_know import BaseCalib
import pandas as pd
import numpy as np 

class BirthRateCalib(BaseCalib):
    """

    """
    
    def __init__(self):
        super().__init__('Birthrate_')
        self.logger.info("")


#INPUTS
#Start with inputs of paper
br_upper_a = 0.39/5
br_lower_a = 0.18/5
delta = 0.00087
phi = 4033.6
nu = 0.18
a = 0.8
cst = 0.022
alpha = 0.0982142178084623
beta = 0.803402073103042
x_zero = [br_upper_a, br_lower_a, delta, phi, nu, a, cst, alpha, beta]


calib = BirthRateCalib()
# bounds = [(-100, 100), (-1e4, 1e4), (-10, 10)]
bounds = [(0.05, 1),(0.005, 0.025), (1e-5, 1), (1000, 1e5), (1e-5, 10), (0.5, 0.95), (0.01,0.02), (1e-4,5), (0,2)]
# calib.eval_all(x_test)
x_opt = calib.optim_variable(x_zero, bounds)
# calib.logger.info(calib.cal_birth_rate_df)
print(x_opt)
# print('calib df', calib.cal_birth_rate_df)
#x_opt = [1.97505881e-01, 2.42204721e-02, 4.94796720e-04, 4.03360023e+03, 2.74706934e-01, 7.89909591e-01, 2.09838226e-02, 5.27058643e-02, 5.22712624e-01]
calib.eval_all(x_opt)
calib.plot_br_gdp()
calib.plot_birth_rate_years()
calib.compute_estimated_birth_rate(x_opt)
calib.plot_estimated_br()
calib.plot_knowledge()


