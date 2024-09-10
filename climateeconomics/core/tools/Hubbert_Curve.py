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

import numpy as np
import pandas as pd

from climateeconomics.glossarycore import GlossaryCore


def compute_Hubbert_regression(past_production, production_years, regression_start, resource_type):

    '''
    Compute Hubbert Regression Curve from past production
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

    tho = 0  # year of resource peak

    # compute of all the possible values of Tho according to Q and P and
    # take the mean values
    for ind in cumulative_sample.index:
        tho = tho + np.log((Q_inf / cumulative_sample.loc[ind, resource_type] - 1) * np.exp(
            cumulative_sample.loc[ind, GlossaryCore.Years] * w)) * (1 / w)
    tho = tho / len(cumulative_sample.index)

    # compute hubbert curve values
    predictable_production = []
    for year in production_years:
        predictable_production += [Q_inf * w * (
                (1 / (np.exp((-(w / 2)) * (tho - year)) + np.exp((w / 2) * (tho - year)))) ** 2),]

    return predictable_production
