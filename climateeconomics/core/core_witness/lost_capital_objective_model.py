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
import numpy as np
import pandas as pd
from sos_trades_core.tools.base_functions.exp_min import compute_func_with_exp_min


class LostCapitalObjective():
    '''
    Used to compute lost capital objective for WITNESS optimization
    '''

    def __init__(self, param):
        '''
        Constructor
        '''
        self.param = param
        self.set_data()
        self.lost_capital_objective = np.array([0.0])
        self.lost_capital_df = None
        self.techno_capital_df = None

    def set_data(self):
        self.year_start = self.param['year_start']
        self.year_end = self.param['year_end']
        self.lost_capital_obj_ref = self.param['lost_capital_obj_ref']
        self.lost_capital_limit = self.param['lost_capital_limit']


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
        Compute the sum of lost_capitals
        """
        self.create_year_range()

        self.lost_capital_df = self.agreggate_and_compute_sum(
            'lost_capital', inputs_dict)

        self.techno_capital_df = self.agreggate_and_compute_sum(
            'techno_capital', inputs_dict)
        self.compute_objective()

    def agreggate_and_compute_sum(self, name, inputs_dict):
        '''
        Aggregate each variable that ends with name in a dataframe and compute the sum of each column
        '''
        name_df_list = [value.drop(
            ['years'], axis=1) for key, value in inputs_dict.items() if key.endswith(name)]

        lost_capital_df = pd.DataFrame({'years': self.years_range})

        if len(name_df_list) != 0:
            lost_capital_df_concat = pd.concat(name_df_list, axis=1)
            name_sum = 'Sum of ' + name.replace('_', ' ')

            lost_capital_df[name_sum] = lost_capital_df_concat.sum(
                axis=1)

            lost_capital_df = pd.concat(
                [lost_capital_df, lost_capital_df_concat], axis=1)

        return lost_capital_df

    def compute_objective(self):
        '''
        Compute objective
        '''
        if 'Sum of lost capital' in self.lost_capital_df:
            self.lost_capital_objective = np.asarray(
                [self.lost_capital_df['Sum of lost capital'].sum()]) / self.lost_capital_obj_ref / self.delta_years

    def get_objective(self):
        '''
        Get lost capital objective
        '''
        return self.lost_capital_objective



    def get_lost_capital_df(self):
        '''
        Get lost capital dataframe with all lost capitals
        '''
        return self.lost_capital_df

    def get_techno_capital_df(self):
        '''
        Get techno capital dataframe with all lost capitals
        '''
        return self.techno_capital_df
