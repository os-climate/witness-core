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
from climateeconomics.core.tools.ClimateEconomicsStudyManager import ClimateEconomicsStudyManager



class Study(StudyManager):

    def __init__(self, year_start = 2020, year_end = 2280, time_step=1, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)
        self.study_name = 'usecase'
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step

    def setup_usecase(self):

        setup_data_list = []
        

        data_dir_resource = join(
            dirname(dirname(dirname(dirname(dirname(dirname(__file__)))))), 'tests', 'data')
        resources_demand = pd.read_csv(
            join(data_dir_resource, 'all_demand_from_energy_mix.csv'))

        

        config_data = {f'{self.study_name}.resources_demand': resources_demand,
                       f'{self.study_name}.year_start': self.year_start,
                       f'{self.study_name}.year_end': self.year_end}

        setup_data_list.append(config_data)


        return setup_data_list



if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()
    ppf = PostProcessingFactory()
    for disc in uc_cls.execution_engine.root_process.sos_disciplines:
        filters = ppf.get_post_processing_filters_by_discipline(
            disc)
        graph_list = ppf.get_post_processing_by_discipline(
            disc, filters, as_json=False)

        # for graph in graph_list:
        #     graph.to_plotly().show()