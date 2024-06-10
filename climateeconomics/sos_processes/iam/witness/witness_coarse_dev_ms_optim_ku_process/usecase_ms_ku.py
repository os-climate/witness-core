"""
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

"""

from os.path import dirname, join

import pandas as pd

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_ms_story_telling.usecase_witness_ms_mda import (
    Study as uc_ms_mda,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_story_telling_optim_process.usecase_2_optim_story_telling import (
    Study as Study1,
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

    def __init__(self, bspline=False, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), "data")

    def setup_usecase(self, study_folder_path=None):

        scatter_scenario = "optimization scenarios"

        scenario_dict = {
            uc_ms_mda.USECASE2: Study1,
            uc_ms_mda.USECASE2B: Study2,
            uc_ms_mda.USECASE4: Study3,
            uc_ms_mda.USECASE7: Study4,
        }

        scenario_df = pd.DataFrame(
            {"selected_scenario": [True] * len(scenario_dict), "scenario_name": list(scenario_dict.keys())}
        )
        values_dict = {
            f"{self.study_name}.{scatter_scenario}.samples_df": scenario_df,
            f"{self.study_name}.n_subcouplings_parallel": min(
                16, len(scenario_df.loc[scenario_df["selected_scenario"] == True])
            ),
        }

        for scenario_name, studyClass in scenario_dict.items():
            scenarioUseCase = studyClass(execution_engine=self.execution_engine)
            scenarioUseCase.study_name = f"{self.study_name}.{scatter_scenario}.{scenario_name}"
            scenarioData = scenarioUseCase.setup_usecase()
            scenarioDatadict = {}
            for data in scenarioData:
                scenarioDatadict.update(data)
            values_dict.update(scenarioDatadict)

        values_dict.update(
            {
                f"{self.study_name}.{scatter_scenario}.{scenario_name}.WITNESS_MDO.max_iter": 400
                for scenario_name in scenario_dict.keys()
            }
        )
        values_dict.update(
            {
                f"{self.study_name}.{scatter_scenario}.{scenario_name}.WITNESS_MDO.WITNESS_Eval.sub_mda_class": "MDAGaussSeidel"
                for scenario_name in scenario_dict.keys()
            }
        )

        return values_dict


if "__main__" == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
