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
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_1_fossil_only_no_damage_low_tax import (
    Study as Study1,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_2_fossil_only_damage_high_tax import (
    Study as Study2,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_3_no_ccs_damage_high_tax import (
    Study as Study3,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_4_all_in_damage_high_tax import (
    Study as Study4,
)


class Study(ClimateEconomicsStudyManager):
    UC1 = "- Damage, - Tax"
    UC2 = "+ Damage, + Tax, Fossil only"
    UC3 = "+ Damage, + Tax, No CCUS"
    UC4 = "+ Damage, + Tax, All technos"
    UC5 = "+ Damage, + Tax, No renewables"

    def __init__(self, bspline=False, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')

    def setup_usecase(self, study_folder_path=None):

        scatter_scenario = 'optimization scenarios'

        scenario_dict = {
            self.UC1: Study1,
            self.UC2: Study2,
            self.UC3: Study3,
            self.UC4: Study4,
            # self.UC5: Study5 # deactivate usecase 5 as it is the same as the one with no CCUS
        }

        scenario_df = pd.DataFrame({'selected_scenario': [True] * len(scenario_dict) ,'scenario_name': list(scenario_dict.keys())})
        values_dict = {
            f'{self.study_name}.{scatter_scenario}.samples_df': scenario_df,
            f'{self.study_name}.n_subcouplings_parallel': min(16, len(scenario_df.loc[scenario_df['selected_scenario']]))
        }

        for scenario_name, studyClass in scenario_dict.items():
            scenarioUseCase = studyClass(execution_engine=self.execution_engine)
            scenarioUseCase.study_name = f'{self.study_name}.{scatter_scenario}.{scenario_name}'
            scenarioData = scenarioUseCase.setup_usecase()
            values_dict.update(scenarioData)

        # override infos
        values_dict.update({f"{self.study_name}.{scatter_scenario}.{scenario_name}.WITNESS_MDO.max_iter": 600 for scenario_name in scenario_dict.keys()})

        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
