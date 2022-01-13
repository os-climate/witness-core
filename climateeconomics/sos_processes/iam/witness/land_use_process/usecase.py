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

from pathlib import Path
from os.path import join, dirname
from numpy import asarray, arange, array
import pandas as pd
import numpy as np
from sos_trades_core.execution_engine.func_manager.func_manager import FunctionManager
from sos_trades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc

INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
OBJECTIVE = FunctionManagerDisc.OBJECTIVE
AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM


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


class Study(StudyManager):

    def __init__(self, year_start=2020, year_end=2100, time_step=1, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)
        self.study_name = 'usecase'
        self.landuse_name = '.Land_Use'
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        livestock_usage_factor_df = None
        self.nb_poles = 8

    def setup_usecase(self):

        setup_data_list = []
        # private values economics operator model
        landuse_input = {}
        landuse_input[self.study_name + '.year_start'] = self.year_start
        landuse_input[self.study_name + '.year_end'] = self.year_end

        landuse_input[self.study_name + self.landuse_name +
                      '.crop_land_use_per_capita'] = 0.21
        landuse_input[self.study_name + self.landuse_name +
                      '.livestock_land_use_per_capita'] = 0.42

        year_range = self.year_end - self.year_start
        years = arange(self.year_start, self.year_end + 1, 1)
        # get population from csv file
        # get file from the data folder 3 folder up.
        global_data_dir = join(Path(__file__).parents[4], 'data')
        population_df = pd.read_csv(
            join(global_data_dir, 'population_df.csv'))
        population_df.index = population_df['years']

        landuse_input[self.study_name +
                      '.population_df'] = population_df

        data_dir = join(
            dirname(dirname(dirname(dirname(dirname(__file__))))), 'tests', 'data')
        land_demand_df = pd.read_csv(
            join(data_dir, 'land_demand.csv'))
        # cut land_demand_df to respect years of study case
        land_demand_df = land_demand_df.loc[land_demand_df['years']
                                            >= self.year_start]
        land_demand_df = land_demand_df.loc[land_demand_df['years']
                                            <= self.year_end]

        landuse_input[self.study_name +
                      '.land_demand_df'] = land_demand_df

        # meat food df : percentage of livestock surface used : quick fix for
        # constraint
        percentage = np.array(np.linspace(90, 50, year_range + 1))
        livestock_usage_factor_df = pd.DataFrame(
            {'years': years, 'percentage': percentage})
        livestock_usage_factor_df.index = years
        self.livestock_usage_factor_df = livestock_usage_factor_df

        landuse_input[self.study_name +
                      '.livestock_usage_factor_df'] = livestock_usage_factor_df

        setup_data_list.append(landuse_input)

        livestock_usage_factor_ctrl = np.linspace(90.0, 50.0, self.nb_poles)

        design_space_ctrl_dict = {}
        design_space_ctrl_dict['livestock_usage_factor_ctrl'] = livestock_usage_factor_ctrl

        design_space_ctrl = pd.DataFrame(design_space_ctrl_dict)
        self.design_space_ctrl = design_space_ctrl
        self.setup_design_space_ctrl()
        return setup_data_list

    def setup_initial_design_variable(self):

        init_design_var_df = pd.DataFrame(
            columns=['percentage'], index=arange(self.year_start, self.year_end + 1, self.time_step))

        init_design_var_df['percentage'] = self.livestock_usage_factor_df['percentage']

        return init_design_var_df

    def setup_design_space(self):
            #-- energy optimization inputs
            # Design Space
        dim_a = self.nb_poles
        lbnd1 = [0.0] * dim_a
        ubnd1 = [100.0] * dim_a

        # Design variables:
        self.update_dspace_dict_with(
            'livestock_usage_factor_array', self.design_space_ctrl['livestock_usage_factor_ctrl'].values, lbnd1, ubnd1)

        #dspace = DataFrame(ddict)

    def setup_design_space_ctrl(self):
            #-- energy optimization inputs
            # Design Space
        header = ['variable', 'value', 'lower_bnd', 'upper_bnd']
        ddict = dict((h, []) for h in header)
        ddict['dspace_size'] = 0

        dim_a = self.nb_poles
        lbnd1 = [0.0] * dim_a
        ubnd1 = [100.0] * dim_a
        # Design variables:
        self.update_dspace_dict_with(
            'livestock_usage_factor_array',
            self.design_space_ctrl['livestock_usage_factor_ctrl'].values, lbnd1, ubnd1)

        #dspace = DataFrame(ddict)

        # return ddict


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.set_debug_mode()
    uc_cls.run()

    ppf = PostProcessingFactory()
    for disc in uc_cls.execution_engine.root_process.sos_disciplines:
        filters = ppf.get_post_processing_filters_by_discipline(
            disc)
        graph_list = ppf.get_post_processing_by_discipline(
            disc, filters, as_json=False)

        for graph in graph_list:
            graph.to_plotly().show()
