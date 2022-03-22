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
        self.create_dataframe()
        self.lost_capital_objective = np.array([0.0])

    def set_data(self):
        self.year_start = self.param['year_start']
        self.year_end = self.param['year_end']
        self.lost_capital_obj_ref = self.param['lost_capital_obj_ref']
        self.energy_list = self.param['energy_list']

    def create_dataframe(self):
        '''
        Create the dataframe and fill it with values at year_start
        '''
        years_range = np.arange(
            self.year_start,
            self.year_end + 1)

        self.lost_capital_df = pd.DataFrame(
            columns=['years', 'Sum of lost capital'])
        self.lost_capital_df['years'] = years_range

    def compute(self, inputs_dict):
        """
        Compute the sum of lost_capitals
        """
        self.create_dataframe()
        lost_capitals_df_list = [value.drop(
            ['years'], axis=1) for key, value in inputs_dict.items() if key.endswith('lost_capital')]

        lost_capital_df = pd.concat(lost_capitals_df_list, axis=1)
        self.lost_capital_df['Sum of lost capital'] = lost_capital_df.sum(
            axis=1)

        self.lost_capital_df = pd.concat(
            [self.lost_capital_df, lost_capital_df], axis=1)
        self.compute_objective()

    def compute_objective(self):
        '''
        Compute objective
        '''
        self.lost_capital_objective = np.asarray(
            [self.lost_capital_df['Sum of lost capital'].sum()]) / self.lost_capital_obj_ref

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
