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
from pandas import DataFrame
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

        self.energy_list = inputs_dict['energy_list']
        self.ccs_list = inputs_dict['ccs_list']
        self.livestock_usage = inputs_dict['livestock_usage']
        self.technology_dict = {
            energy: inputs_dict[f'{energy}.technologies_list'] for energy in self.energy_list + self.ccs_list}
        self.output_dict = {}

        self.bspline_dict = {}
        self.dspace = inputs_dict['design_space']

    def configure(self, inputs_dict):
        '''
        Configure with inputs_dict from the discipline
        '''
        self.output_dict = {}
        if self.livestock_usage:
            list_ctrl = ['livestock_usage_factor_array']
        else:
            list_ctrl = []

        list_ctrl.extend(
            [key for key in inputs_dict if key.endswith('_array_mix')])

        years = arange(self.year_start, self.year_end + 1, self.time_step)

        list_t_years = np.linspace(0.0, 1.0, len(years))

        for full_elem in list_ctrl:
            elem = full_elem.split('.')[-1]
            l_activated = self.dspace.loc[self.dspace[self.VARIABLES]
                                          == elem, self.ACTIVATED_ELEM_LIST].to_list()[0]
            value_dv = self.dspace.loc[self.dspace[self.VARIABLES]
                                       == elem, self.VALUE].to_list()[0]
            elem_val = inputs_dict[full_elem]
            index_false = None
            if not all(l_activated):
                index_false = l_activated.index(False)
                elem_val = list(elem_val)
                elem_val.insert(index_false, value_dv[index_false])
                elem_val = np.asarray(elem_val)

            if len(inputs_dict[full_elem]) == len(years):
                self.bspline_dict[full_elem] = {
                    'bspline': None, 'eval_t': inputs_dict[full_elem], 'b_array': np.identity(len(years))}
            else:
                bspline = BSpline(n_poles=len(elem_val))
                bspline.set_ctrl_pts(elem_val)
                eval_t, b_array = bspline.eval_list_t(list_t_years)
                b_array = bspline.update_b_array(b_array, index_false)

                self.bspline_dict[full_elem] = {
                    'bspline': bspline, 'eval_t': eval_t, 'b_array': b_array}
        #######

        if self.livestock_usage:
            livestock_usage_factor_df = DataFrame(
                {'years': years, 'percentage': self.bspline_dict['livestock_usage_factor_array']['eval_t']}, index=years)
            self.output_dict['livestock_usage_factor_df'] = livestock_usage_factor_df

        dict_mix = {'years': years}

        for energy in self.energy_list + self.ccs_list:
            energy_wo_dot = energy.replace('.', '_')
            dict_mix.update(
                {f'{energy}.{techno}': self.bspline_dict[f"{energy}.{techno}.{energy_wo_dot}_{techno.replace('.', '_')}_array_mix"]['eval_t'] for techno in self.technology_dict[energy]})

        self.output_dict['invest_mix'] = DataFrame(
            dict_mix, index=years)
