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
from pandas import read_csv
import pandas as pd
import numpy as np
from sos_trades_core.execution_engine.func_manager.func_manager import FunctionManager
from sos_trades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc

INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
OBJECTIVE = FunctionManagerDisc.OBJECTIVE
AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM


class Study(StudyManager):

    def __init__(self, year_start=2020, year_end=2100, time_step=1, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)
        self.study_name = 'usecase'
        self.landuse_name = '.Population'
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step

    def setup_usecase(self):

        setup_data_list = []
        nb_per = round(
            (self.year_end - self.year_start) / self.time_step + 1)
        years = arange(self.year_start, self.year_end + 1, self.time_step)
        global_data_dir = join(Path(__file__).parents[4], 'data')
        # private values economics operator model
        population_input = {}
        population_input[self.study_name + '.year_start'] = self.year_start
        population_input[self.study_name + '.year_end'] = self.year_end
        population_input[self.study_name + '.time_step'] = self.time_step

        gdp_year_start = 130.187
        gdp_serie = []
        gdp_serie.append(gdp_year_start)
        for year in np.arange(1, nb_per):
            gdp_serie.append(gdp_serie[year - 1] * 1.02)

        economics_df_y = pd.DataFrame(
            {'years': years, 'output_net_of_d': gdp_serie})
        economics_df_y.index = years
        temperature_df_all = read_csv(
            join(global_data_dir, 'temperature_data_onestep.csv'))

        population_input[self.study_name +
                         '.economics_df'] = economics_df_y
        population_input[self.study_name +
                         '.temperature_df'] = temperature_df_all

        setup_data_list.append(population_input)

        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    # uc_cls.execution_engine.set_debug_mode()
    uc_cls.run()

    ppf = PostProcessingFactory()
    for disc in uc_cls.execution_engine.root_process.sos_disciplines:
        filters = ppf.get_post_processing_filters_by_discipline(
            disc)
        graph_list = ppf.get_post_processing_by_discipline(
            disc, filters, as_json=False)

        for graph in graph_list:
            graph.to_plotly().show()
