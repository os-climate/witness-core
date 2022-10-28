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
from climateeconomics.core.core_witness.calibration.calibration_prodfunction.meauxmorere_functionCD import MeauxMorerePibCD

#To modify code of projection: base calib2022 
Test = MeauxMorerePibCD() 
#Projection
x_opt = [0.01902798, 0.118197,  0.02956916, 0.64780347]
x_opt = [0.01917721, 0.14104998, 0.03472088, 0.7]
Test.eval_projection(2050, x_opt, scenario = 'high')   

# Test.plot_productivity()
# Test.plot_energy()
# Test.plot_workforce()