'''
Copyright 2022 Airbus SAS
Modifications on 27/11/2023-2024/06/24 Copyright 2023 Capgemini

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
from os.path import dirname, join

import numpy as np
import pandas as pd
from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.post_processing.post_processing_factory import (
    PostProcessingFactory,
)

from climateeconomics.sos_processes.iam.witness.witness_optim_process.usecase_witness_optim import (
    Study as witness_optim_usecase,
)


class Study(StudyManager):

    def __init__(self, bspline=False, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')
        self.test_post_procs = False

    def setup_usecase(self, study_folder_path=None):
        witness_ms_usecase = witness_optim_usecase(
            execution_engine=self.execution_engine)

        self.scatter_scenario = 'optimization scenarios'
        # Set public values at a specific namespace
        witness_ms_usecase.study_name = f'{self.study_name}.{self.scatter_scenario}'

        values_dict = {}
        scenario_list = []
        alpha_list = np.linspace(0, 100, 3, endpoint=True) / 100.0
        for alpha_i in alpha_list:
            scenario_i = 'scenario_\u03B1=%.2f' % alpha_i
            scenario_i = scenario_i.replace('.', ',')
            scenario_list.append(scenario_i)
            values_dict[f'{self.study_name}.{self.scatter_scenario}.{scenario_i}.{witness_ms_usecase.optim_name}.{witness_ms_usecase.coupling_name}.{witness_ms_usecase.extra_name}.alpha'] = alpha_i

#         values_dict[f'{self.study_name}.epsilon0'] = 1.0
        values_dict[f'{self.study_name}.n_subcouplings_parallel'] = 3
        len_scenarios = len(scenario_list)
        scenario_df = pd.DataFrame({'selected_scenario': [True] * len_scenarios ,'scenario_name': scenario_list})

        values_dict[f'{self.study_name}.{self.scatter_scenario}.samples_df'] = scenario_df

        for scenario in scenario_list:
            scenarioUseCase = witness_optim_usecase(
                bspline=self.bspline, execution_engine=self.execution_engine)
            scenarioUseCase.optim_name = f'{scenario}.{scenarioUseCase.optim_name}'
            scenarioUseCase.study_name = witness_ms_usecase.study_name
            scenarioData = scenarioUseCase.setup_usecase()

            for dict_data in scenarioData:
                values_dict.update(dict_data)

        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()

    ppf = PostProcessingFactory()
    filters = ppf.get_post_processing_filters_by_namespace(
        uc_cls.execution_engine, f'{uc_cls.study_name}.Post-processing')
    graph_list = ppf.get_post_processing_by_namespace(uc_cls.execution_engine, f'{uc_cls.study_name}.Post-processing',
                                                      filters, as_json=False)

#    for graph in graph_list:
#        graph.to_plotly().show()
