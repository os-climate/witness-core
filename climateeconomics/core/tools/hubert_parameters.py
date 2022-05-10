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

from os.path import join, dirname

from scipy.fftpack import diff
from climateeconomics.core.core_resources.models.copper_resource.copper_resource_model import CopperResourceModel
import numpy as np
import pandas as pd


resource_name = CopperResourceModel.resource_name
Q_inf_th = 2851.691 #reserve underground + Q(2020)
past_production = pd.read_csv(join(dirname(__file__), f'../resources_data/{resource_name}_production_data.csv'))
production_start = 1925
production_years = np.arange(production_start, 2101)
past_production_years = np.arange(production_start, 2021)


def compute_Hubbert_parameters(past_production, production_years, regression_start, resource_type):
    
    '''
    Compute Hubbert parameters from past production
    '''
    # initialization
    past_production_years=past_production['years']
    cumulative_production = pd.DataFrame(
        {'years': past_production_years, f'{resource_type}': np.zeros(len(past_production_years))})
    ratio_P_by_Q = pd.DataFrame({'years': past_production_years, f'{resource_type}': np.zeros(len(past_production_years))})
    
    # Cf documentation for the hubbert curve computing
    Q = 0  # Q is the cumulative production at one precise year

    # compute cumulative production, the production, and the ratio
    # dataframe for each year
    for i_year, pp_year in enumerate(past_production_years):
        P = past_production.loc[i_year, resource_type]
        cumulative_production.loc[i_year,
                                       resource_type] = Q + P
        Q = cumulative_production.loc[i_year, resource_type]
        ratio_P_by_Q.loc[i_year, resource_type] = P / Q
    
    # keep only the part you want to make a regression on
    cumulative_sample = cumulative_production.loc[
        cumulative_production['years'] >= regression_start]
    
    

    ratio_sample = ratio_P_by_Q.loc[ratio_P_by_Q['years'] >= regression_start]

    fit = np.polyfit(
        cumulative_sample[resource_type], ratio_sample[resource_type], 1)

    w = fit[1]  # imaginary frequency
    

    # sum of the available and recoverable reserve (predict by Hubbert
    # model from the start of the exploitation to the end)
    Q_inf = -1 * (w / fit[0])

    return Q_inf

year_regression = 1925

difference = 1000

#Goes through all the past years and returns the year of the regression
for evolving_year in past_production_years :
    Q_inf = compute_Hubbert_parameters(past_production, production_years, evolving_year, 'copper')
    if abs(Q_inf_th - Q_inf) < difference :
        difference = abs (Q_inf_th - Q_inf)
        year_regression = evolving_year
print("l'année de régression est : ")
print (year_regression)

print("et la différence entre Q_inf et Q_inf_th est de ")
print (difference)