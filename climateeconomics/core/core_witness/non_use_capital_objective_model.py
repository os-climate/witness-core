'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/21-2023/11/03 Copyright 2023 Capgemini

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
from energy_models.glossaryenergy import GlossaryEnergy


class NonUseCapitalObjective():
    '''
    Used to compute non use capital objective for WITNESS optimization
    '''

    def __init__(self, param):
        '''
        Constructor
        '''
        self.param = param
        self.set_data()
        self.non_use_capital_objective = np.array([0.0])
        self.non_use_capital_df = None
        self.techno_capital_df = None
        self.non_use_capital_cons = np.array([0.0])
        self.forest_lost_capital_cons = np.array([0.0])

    def set_data(self):
        self.year_start = self.param[GlossaryCore.YearStart]
        self.year_end = self.param[GlossaryCore.YearEnd]
        self.non_use_capital_obj_ref = self.param['non_use_capital_obj_ref']
        self.non_use_capital_cons_limit = self.param['non_use_capital_cons_limit']
        self.non_use_capital_cons_ref = self.param['non_use_capital_cons_ref']

        self.forest_lost_capital_cons_limit = self.param['forest_lost_capital_cons_limit']
        self.forest_lost_capital_cons_ref = self.param['forest_lost_capital_cons_ref']
        self.forest_lost_capital = self.param['forest_lost_capital']

    def create_year_range(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        self.years_range = np.arange(
            self.year_start,
            self.year_end + 1)
        self.delta_years = len(self.years_range)

    def compute(self, inputs_dict):
        """
        Compute the sum of non_use_capitals
        """
        self.create_year_range()

        self.non_use_capital_df = self.agreggate_and_compute_sum(
            'non_use_capital', inputs_dict)

        self.techno_capital_df = self.agreggate_and_compute_sum(
            GlossaryEnergy.TechnoCapitalValue, inputs_dict)

        self.forest_lost_capital = inputs_dict['forest_lost_capital']

        self.compute_objective()
        self.compute_constraint()

    def agreggate_and_compute_sum(self, name, inputs_dict):
        '''
        Aggregate each variable that ends with name in a dataframe and compute the sum of each column
        '''
        name_df_list = [value.drop(
            [GlossaryCore.Years], axis=1) for key, value in inputs_dict.items() if key.endswith(name)]

        non_use_capital_df = pd.DataFrame({GlossaryCore.Years: self.years_range})

        if len(name_df_list) != 0:
            non_use_capital_df_concat = pd.concat(name_df_list, axis=1)
            name_sum = 'Sum of ' + name.replace('_', ' ')

            non_use_capital_df[name_sum] = non_use_capital_df_concat.sum(
                axis=1)

            non_use_capital_df = pd.concat(
                [non_use_capital_df, non_use_capital_df_concat], axis=1)

        return non_use_capital_df

    def compute_objective(self):
        '''
        Compute objective
        '''
        if 'Sum of non use capital' in self.non_use_capital_df:
            self.non_use_capital_objective_wo_ponderation = np.asarray(
                [self.non_use_capital_df['Sum of non use capital'].sum()]) / self.delta_years
            self.non_use_capital_objective = self.non_use_capital_objective_wo_ponderation / \
                                             self.non_use_capital_obj_ref

    def compute_constraint(self):
        '''
        Compute constraint
        '''
        if 'Sum of non use capital' in self.non_use_capital_df:
            self.non_use_capital_cons = (
                                                self.non_use_capital_cons_limit - self.non_use_capital_objective_wo_ponderation) / self.non_use_capital_cons_ref

        reforestation_lost_capital_wo_ponderation = np.asarray(
            [self.forest_lost_capital['reforestation'].sum()]) / self.delta_years
        managed_wood_lost_capital_wo_ponderation = np.asarray(
            [self.forest_lost_capital['managed_wood'].sum()]) / self.delta_years
        deforestation_lost_capital_wo_ponderation = np.asarray(
            [self.forest_lost_capital['deforestation'].sum()]) / self.delta_years
        forest_lost_capital_wo_ponderation = reforestation_lost_capital_wo_ponderation + \
                                             managed_wood_lost_capital_wo_ponderation + \
                                             deforestation_lost_capital_wo_ponderation
        self.forest_lost_capital_cons = (
                                                self.forest_lost_capital_cons_limit - forest_lost_capital_wo_ponderation) / self.forest_lost_capital_cons_ref

    def get_energy_capital_trillion_dollars(self):
        '''
        Get energy capital dataframe in trillion dollars
        The sum is in G$ (1e9 $)
        '''
        if 'Sum of techno capital' in self.techno_capital_df:
            sum_techno_capital = self.techno_capital_df['Sum of techno capital'].values / 1e3
        else:
            sum_techno_capital = 0.0
        energy_capital_df = pd.DataFrame({GlossaryCore.Years: self.years_range,
                                          GlossaryCore.Capital: sum_techno_capital})

        return energy_capital_df
