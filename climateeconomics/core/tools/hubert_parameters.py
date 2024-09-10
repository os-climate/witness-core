'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2023/11/03 Copyright 2023 Capgemini

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

from os.path import dirname, join

import numpy as np
import pandas as pd

from climateeconomics.core.core_resources.models.copper_resource.copper_resource_model import (
    CopperResourceModel,
)
from climateeconomics.glossarycore import GlossaryCore

resource_name = CopperResourceModel.resource_name  # CopperResourceModel.resource_name  PlatinumResourceModel.resource_name
known_reserves = 2100  # 2100    0.0354
undiscovered_resources = 3500  # 3500
future_discovery = 0.5
hypot_reservers = known_reserves + future_discovery * undiscovered_resources
Q_2020 = 751.691  # 751.691  0.007319767919
Q_inf_th = Q_2020 + hypot_reservers  # Q(2020) + reserve underground
past_production = pd.read_csv(join(dirname(__file__), f'../core_resources/models/resources_data/{resource_name}_production_data.csv'))
production_start = 1925  # 1925   1967
production_years = np.arange(production_start, GlossaryCore.YearEndDefault + 1)
past_production_years = np.arange(production_start, 2021)


def compute_Hubbert_parameters(past_production, production_years, regression_start, resource_type):

    '''
    Compute Hubbert parameters from past production
    '''
    # initialization
    past_production_years = past_production[GlossaryCore.Years]
    cumulative_production = pd.DataFrame(
        {GlossaryCore.Years: past_production_years, f'{resource_type}': np.zeros(len(past_production_years))})
    ratio_P_by_Q = pd.DataFrame({GlossaryCore.Years: past_production_years, f'{resource_type}': np.zeros(len(past_production_years))})

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
        cumulative_production[GlossaryCore.Years] >= regression_start]

    ratio_sample = ratio_P_by_Q.loc[ratio_P_by_Q[GlossaryCore.Years] >= regression_start]

    fit = np.polyfit(
        cumulative_sample[resource_type], ratio_sample[resource_type], 1)

    w = fit[1]  # imaginary frequency

    # sum of the available and recoverable reserve (predict by Hubbert
    # pyworld3 from the start of the exploitation to the end)
    Q_inf = -1 * (w / fit[0])

    return Q_inf


# Loops going through the potential years and selecting optimum regression year ###


# start of the loop, must be the first year in the past_production csv
year_regression = 1925  # 1925  1967

difference = 1000
Q_inf_taken = 5
new_difference = 1000
new_year_regression = 0
end_of_past_production = 0
final_Q_inf = None
# Goes through all the past years and returns the year of the regression
for evolving_year in past_production_years:

    # loop also the last_year of regression
    new_past_production = past_production.loc[past_production[GlossaryCore.Years] <= evolving_year]

    print(evolving_year)
    for year in np.arange(production_start + 1, evolving_year):
        Q_inf = compute_Hubbert_parameters(new_past_production, production_years, year, 'copper')  # 'copper' 'platinum
        if abs(Q_inf_th - Q_inf) < difference:
            difference = abs(Q_inf_th - Q_inf)
            year_regression = year
            Q_inf_taken = Q_inf

    if abs(Q_inf_th - Q_inf_taken) < new_difference:
        new_difference = abs(Q_inf_th - Q_inf_taken)
        new_year_regression = year_regression
        final_Q_inf = Q_inf_taken
        end_of_past_production = evolving_year
if final_Q_inf is None:
    raise Exception("assertion abs(Q_inf_th - Q_inf_taken) < new_difference is not True")
error = ((Q_inf_th - final_Q_inf) / Q_inf_th) * 100


# Graph with past production and hubert curve determined with the optimum parameters

# production = compute_Hubbert_regression(past_production, production_years, new_year_regression, 'copper')

# plt.title('Production according to Hubert vs real production')
# plt.plot(list(production_years), list(production))
# plt.plot(np.arange(production_start, 2021), list(past_production['copper']))
# plt.xlabel(GlossaryCore.Years)
# plt.ylabel('Potential production [Mt]')
# plt.show()


print("l'année de régression est : ")
print(new_year_regression)

print(f"la fin de la régression est {end_of_past_production}")

print("et la différence entre Q_inf_th et Q_inf est de ")
print(new_difference)

print(f"le taux d'erreur est de {error} %")

print(f"celui qu'on veut obtenir est {Q_inf_th} et celui qu'on calcule est de {final_Q_inf} ")
