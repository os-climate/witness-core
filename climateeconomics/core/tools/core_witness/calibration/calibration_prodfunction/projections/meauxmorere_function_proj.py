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
from climateeconomics.core.core_witness.calibration.calibration_prodfunction.meauxmorere_function import MeauxMorerePib



#To modify projection code go to base_calib    
Test = MeauxMorerePib()
#Choose x for projection
x_opt = [0.02148322, 0.16474359, 0.03279633, 0.88139106]   
x_opt = [0.01777792, 0.15720722, 0.02387787, 0.86537018]
#and projection 
Test.eval_projection(2050, x_opt, scenario = 'high')
# Test.plot_productivity()
# Test.plot_energy()
# Test.plot_workforce()
