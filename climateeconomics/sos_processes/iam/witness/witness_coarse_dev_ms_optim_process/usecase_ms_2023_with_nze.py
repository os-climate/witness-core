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
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_3_no_ccs_damage_high_tax import (
    Study as Study3,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_4_all_in_damage_high_tax import (
    Study as Study4,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_2023_nze_2050 import (
    Study as StudyNZE,
)


class Study(ClimateEconomicsStudyManager):
    NO_CCUS = "No CCUS"
    ALL_TECHNOS = "All technos"
    ALL_TECHNOS_NZE = "All technos NZE"

    def __init__(self, year_start=2023, bspline=False, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')
        self.year_start = year_start

    def setup_usecase(self, study_folder_path=None):

        scatter_scenario = 'optimization scenarios'

        scenario_dict = {
            self.NO_CCUS: Study3,
            self.ALL_TECHNOS: Study4,
            self.ALL_TECHNOS_NZE: StudyNZE,
        }

        # changing the tipping point

        scenario_df = pd.DataFrame({'selected_scenario': [True] * len(scenario_dict), 'scenario_name': list(scenario_dict.keys())})
        values_dict = {
            f'{self.study_name}.{scatter_scenario}.samples_df': scenario_df,
            f'{self.study_name}.n_subcouplings_parallel': min(16, len(scenario_df.loc[scenario_df['selected_scenario']]))
        }

        for scenario_name, studyClass in scenario_dict.items():
            scenarioUseCase = studyClass(execution_engine=self.execution_engine, year_start=self.year_start)
            scenarioUseCase.study_name = f'{self.study_name}.{scatter_scenario}.{scenario_name}'
            scenarioData = scenarioUseCase.setup_usecase()
            scenarioDatadict = {}
            scenarioDatadict.update(scenarioData)
            values_dict.update(scenarioDatadict)

        values_dict.update({f"{self.study_name}.{scatter_scenario}.{scenario_name}.WITNESS_MDO.max_iter": 400 for scenario_name in scenario_dict.keys()})
        # update the tipping point value

        values_dict = self.update_dataframes_with_year_star(values_dict=values_dict, year_start=self.year_start)

        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
