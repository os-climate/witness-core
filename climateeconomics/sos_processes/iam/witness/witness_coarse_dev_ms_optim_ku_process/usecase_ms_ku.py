'''
Copyright 2024 Capgemini
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

import pandas as pd

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_ms_story_telling.usecase_witness_ms_mda import (
    Study as uc_ms_mda,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_story_telling_optim_process.usecase_2b_optim_story_telling import (
    Study as Study2,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_story_telling_optim_process.usecase_4_optim_story_telling import (
    Study as Study3,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_story_telling_optim_process.usecase_7_optim_story_telling import (
    Study as Study4,
)


class Study(ClimateEconomicsStudyManager):

    def __init__(self, year_start=2023, filename=__file__, bspline=False, run_usecase=False, execution_engine=None):
        super().__init__(filename, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')
        self.test_post_procs = True
        self.year_start = year_start

        self.scatter_scenario = 'optimization scenarios'

        self.scenario_dict = {
            #uc_ms_mda.USECASE2: Study1,
            uc_ms_mda.USECASE2B: Study2,
            uc_ms_mda.USECASE4: Study3,
            uc_ms_mda.USECASE7: Study4,
        }
    def setup_usecase(self, study_folder_path=None):

        scenario_df = pd.DataFrame({'selected_scenario': [True] * len(self.scenario_dict) ,'scenario_name': list(self.scenario_dict.keys())})
        values_dict = {
            f'{self.study_name}.{self.scatter_scenario}.samples_df': scenario_df,
            f'{self.study_name}.n_subcouplings_parallel': min(16, len(scenario_df.loc[scenario_df['selected_scenario']]))
        }

        for scenario_name, studyClass in self.scenario_dict.items():
            scenarioUseCase = studyClass(execution_engine=self.execution_engine, year_start=self.year_start)
            scenarioUseCase.study_name = f'{self.study_name}.{self.scatter_scenario}.{scenario_name}'
            scenarioData = scenarioUseCase.setup_usecase()
            values_dict.update(scenarioData)

        values_dict.update({f"{self.study_name}.{self.scatter_scenario}.{scenario_name}.WITNESS_MDO.max_iter": 400 for scenario_name in self.scenario_dict.keys()})
        values_dict.update(
            {f"{self.study_name}.{self.scatter_scenario}.{scenario_name}.WITNESS_MDO.WITNESS_Eval.inner_mda_name": "MDAGaussSeidel" for scenario_name in
             self.scenario_dict.keys()})

        values_dict = self.update_dataframes_with_year_star(values_dict=values_dict, year_start=self.year_start)

        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
