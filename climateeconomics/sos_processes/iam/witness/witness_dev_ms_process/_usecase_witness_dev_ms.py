"""
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
"""

from os.path import dirname, join

import numpy as np
from sostrades_core.study_manager.study_manager import StudyManager

from climateeconomics.sos_processes.iam.witness.witness_dev.usecase_witness import (
    Study as witness_dev_usecase,
)


class Study(StudyManager):

    def __init__(self, bspline=False, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), "data")

    def setup_usecase(self, study_folder_path=None):
        witness_ms_usecase = witness_dev_usecase(execution_engine=self.execution_engine)

        self.scatter_scenario = "multiscenario"
        # Set public values at a specific namespace
        witness_ms_usecase.study_name = f"{self.study_name}.{self.scatter_scenario}"

        values_dict = {}
        scenario_list = []
        alpha_list = np.arange(0, 500, 1)
        for alpha_i in alpha_list:
            scenario_i = f"scenario_{alpha_i}"
            scenario_list.append(scenario_i)

        values_dict[f"{self.study_name}.{self.scatter_scenario}.scenario_list"] = scenario_list

        for scenario in scenario_list:
            scenarioUseCase = witness_dev_usecase(bspline=self.bspline, execution_engine=self.execution_engine)
            scenarioUseCase.study_name = witness_ms_usecase.study_name
            scenarioData = scenarioUseCase.setup_usecase()

            for dict_data in scenarioData:
                values_dict.update(dict_data)

        return values_dict


if "__main__" == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
