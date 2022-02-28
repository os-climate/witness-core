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
from numpy import arange
from pandas import DataFrame, Series
from sos_trades_core.tools.bspline.bspline import BSpline

import numpy as np


class Design_var(object):
    """
    Class Design variable
    """
    ACTIVATED_ELEM_LIST = "activated_elem"
    VARIABLES = "variable"
    VALUE = "value"

    def __init__(self, inputs_dict):
        self.year_start = inputs_dict['year_start']
        self.year_end = inputs_dict['year_end']
        self.time_step = inputs_dict['time_step']
        self.output_descriptor = inputs_dict['output_descriptor']
        self.output_dict = {}
        self.bspline_dict = {}
        self.dspace = inputs_dict['design_space']

    def configure(self, inputs_dict):
        '''
        Configure with inputs_dict from the discipline
        '''

        self.output_dict = {}
        list_ctrl = self.output_descriptor.keys()
        years = arange(self.year_start, self.year_end + 1, self.time_step)
        list_t_years = np.linspace(0.0, 1.0, len(years))

        for elem in list_ctrl:
            l_activated = self.dspace.loc[self.dspace[self.VARIABLES]
                                          == elem, self.ACTIVATED_ELEM_LIST].to_list()[0]
            value_dv = self.dspace.loc[self.dspace[self.VARIABLES]
                                       == elem, self.VALUE].to_list()[0]
            elem_val = inputs_dict[elem]
            index_false = None
            if not all(l_activated):
                index_false = l_activated.index(False)
                elem_val = list(elem_val)
                elem_val.insert(index_false, value_dv[index_false])
                elem_val = np.asarray(elem_val)

            if len(inputs_dict[elem]) == len(years):
                self.bspline_dict[elem] = {
                    'bspline': None, 'eval_t': inputs_dict[elem], 'b_array': np.identity(len(years))}
            else:
                bspline = BSpline(n_poles=len(elem_val))
                bspline.set_ctrl_pts(elem_val)
                eval_t, b_array = bspline.eval_list_t(list_t_years)
                b_array = bspline.update_b_array(b_array, index_false)

                self.bspline_dict[elem] = {
                    'bspline': bspline, 'eval_t': eval_t, 'b_array': b_array}
        #######

        # loop over output_descriptor to build output
        for key in self.output_descriptor.keys():
            out_name = self.output_descriptor[key]['out_name']
            out_type = self.output_descriptor[key]['type']

            if out_type == 'array':
                self.output_dict[out_name] = self.bspline_dict[key]['eval_t']
            elif out_type == 'dataframe':
                if self.output_descriptor[key]['out_name'] not in self.output_dict.keys():
                    # init output dataframes with 'years' index
                    self.output_dict[out_name] = DataFrame({'years': years}, index=years)

                col_name = self.output_descriptor[key]['key']
                self.output_dict[out_name][col_name] = self.bspline_dict[key]['eval_t']
            else:
                raise(ValueError('Output type not yet supported'))
