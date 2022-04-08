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

from sos_trades_core.tools.post_processing.post_processing_factory import PostProcessingFactory
from sos_trades_core.study_manager.study_manager import StudyManager
from climateeconomics.core.core_forest.forest_v2 import Forest

from pathlib import Path
from os.path import join, dirname
from numpy import asarray, arange, array
import pandas as pd
import numpy as np
from sos_trades_core.execution_engine.func_manager.func_manager import FunctionManager
from sos_trades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc


def update_dspace_with(dspace_dict, name, value, lower, upper):
    ''' type(value) has to be ndarray
    '''
    if not isinstance(lower, (list, np.ndarray)):
        lower = [lower] * len(value)
    if not isinstance(upper, (list, np.ndarray)):
        upper = [upper] * len(value)
    dspace_dict['variable'].append(name)
    dspace_dict['value'].append(value.tolist())
    dspace_dict['lower_bnd'].append(lower)
    dspace_dict['upper_bnd'].append(upper)
    dspace_dict['dspace_size'] += len(value)


def update_dspace_dict_with(dspace_dict, name, value, lower, upper, activated_elem=None, enable_variable=True):
    if not isinstance(lower, (list, np.ndarray)):
        lower = [lower] * len(value)
    if not isinstance(upper, (list, np.ndarray)):
        upper = [upper] * len(value)

    if activated_elem is None:
        activated_elem = [True] * len(value)
    dspace_dict[name] = {'value': value,
                         'lower_bnd': lower, 'upper_bnd': upper, 'enable_variable': enable_variable, 'activated_elem': activated_elem}

    dspace_dict['dspace_size'] += len(value)


class Study(StudyManager):

    def __init__(self, year_start=2020, year_end=2100, time_step=1, name='.Forest', execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)
        self.study_name = 'usecase'
        self.forest_name = name
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.nb_poles = 8

    def setup_usecase(self):

        setup_data_list = []

        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1

        energy_list = []
        ccs_list = []
        agriculture_techno_list = ['Forest']
        non_use_capital_obj_ref = 10
        non_use_capital_limit = 300
        gamma = 0.5
        non_use_capital_ref = pd.DataFrame({'years': np.arange(self.year_start, self.year_end + 1),
                                            'Forest': 3})
        # values of model
        capital_input = {}
        capital_input[self.study_name + '.year_start'] = self.year_start
        capital_input[self.study_name + '.year_end'] = self.year_end

        capital_input[self.study_name +
                      '.energy_list'] = energy_list
        capital_input[self.study_name +
                      '.ccs_list'] = ccs_list
        capital_input[self.study_name +
                      '.Agriculture.Forest.agri_capital_techno_list'] = agriculture_techno_list
        capital_input[self.study_name +
                      '.non_use_capital_obj_ref'] = non_use_capital_obj_ref
        capital_input[self.study_name +
                      '.non_use_capital_limit'] = non_use_capital_limit
        capital_input[self.study_name +
                      '.gamma'] = gamma
        capital_input[self.study_name +
                      '.Agriculture.Forest.non_use_capital'] = non_use_capital_ref
        capital_input[self.study_name +
                      '.Agriculture.Forest.techno_capital'] = non_use_capital_ref

        setup_data_list.append(capital_input)

        deforestation_surface_ctrl = np.linspace(10.0, 5.0, self.nb_poles)

        return setup_data_list

    def setup_initial_design_variable(self):

        init_design_var_df = pd.DataFrame(
            columns=['deforested_surface', 'forest_investment'], index=arange(self.year_start, self.year_end + 1, self.time_step))

        init_design_var_df['deforested_surface'] = self.deforestation_surface['deforested_surface']
        init_design_var_df['forest_investment'] = self.forest_invest_df['forest_investment']

        return init_design_var_df

    def setup_design_space(self):
        #-- energy optimization inputs
        # Design Space
        dim_a = len(
            self.deforestation_surface['deforested_surface'].values)
        lbnd1 = [0.0] * dim_a
        ubnd1 = [100.0] * dim_a

        # Design variables:
        self.update_dspace_dict_with(
            'deforested_surface_array', self.deforestation_surface['deforested_surface'].values, lbnd1, ubnd1)
        self.update_dspace_dict_with(
            'forest_investment_array', self.forest_invest_df['forest_investment'].values, lbnd1, ubnd1)

    def setup_design_space_ctrl_new(self):
        # Design Space
        # header = ['variable', 'value', 'lower_bnd', 'upper_bnd']
        ddict = {}
        ddict['dspace_size'] = 0

        # Design variables:
        update_dspace_dict_with(ddict, 'deforested_surface_ctrl',
                                list(self.design_space_ctrl['deforested_surface_ctrl'].values), [0.0] * self.nb_poles, [100.0] * self.nb_poles, activated_elem=[True, True, True, True, True, True, True, True])

        update_dspace_dict_with(ddict, 'forest_investment_array_mix',
                                list(self.design_space_ctrl['forest_investment_array_mix'].values), [1.0e-6] * self.nb_poles, [3000.0] * self.nb_poles, activated_elem=[True, True, True, True, True, True, True, True])
        return ddict


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    # uc_cls.execution_engine.set_debug_mode()
    uc_cls.run()

    # ppf = PostProcessingFactory()
    # for disc in uc_cls.execution_engine.root_process.sos_disciplines:
    #     filters = ppf.get_post_processing_filters_by_discipline(
    #         disc)
    #     graph_list = ppf.get_post_processing_by_discipline(
    #         disc, filters, as_json=False)

    #     for graph in graph_list:
    #         graph.to_plotly().show()
