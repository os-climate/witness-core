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

from climateeconomics.core.core_witness.calibration.calibration_pop.base_calib_pop import BaseCalib
import pandas as pd
import numpy as np 

class BirthRateCalib(BaseCalib):
    """
    Calibrate the birth rate using formula from: 
    McIsaac, F., 2020. A Representation of the World Population Dynamics for Integrated Assessment Models. 
    Environmental Modeling & Assessment, 25(5), pp.611-632.
    """
    
    def __init__(self):
        super().__init__('Birthrate_lower0022')
        self.logger.info("")


#INPUTS
#Start with inputs of paper
br_upper_a = 0.39/5
br_lower_a = 0.18/5
delta = 0.00087
phi = 4033.6
nu = 0.18
x_zero = [8.0e-02, 0.022, 4.40867219e-04, 4.03359979e+03, 2.22178157e-01 ]

x_zero = [br_upper_a, br_lower_a, delta, phi, nu]

calib = BirthRateCalib()
bounds = [(0.005, 0.08),(0.005, 0.022), (1e-5, 1), (1000, 1e5), (1e-5, 5)]
# x_test = [0.8, 0.05, 1e-5, 4033, 1e-5]
# x_test = [8.00000000e-01, 5.00000000e-01, 1.00000000e-05,4.03360153e03, 1.00000000e-05]
# calib.eval_all(x_test)
x_opt = calib.optim_variable(x_zero, bounds)
calib.logger.info(calib.cal_birth_rate_df)
print(x_opt)
print('calib df', calib.cal_birth_rate_df)
calib.eval_all(x_opt)
calib.plot_br_gdp()
