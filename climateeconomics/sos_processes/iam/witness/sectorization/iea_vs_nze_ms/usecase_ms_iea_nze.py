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
from climateeconomics.sos_processes.iam.witness.sectorization.iea_vs_nze.usecase_witness_full_iea_nze import (
    Study as StudyIEANZE,
)


class Study(ClimateEconomicsStudyManager):
    TIPPING_POINT_LIST = [6, 4.5, 3.5]
    def __init__(self, bspline=True, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')

    def setup_usecase(self, study_folder_path=None):

        scatter_scenario = 'scenarios'

        scenario_dict = {
          f'Tipping point {str(tp).replace('.', '_')} degrés'  : {'tp_a3':tp} for tp in self.TIPPING_POINT_LIST
        }

        scenario_df = pd.DataFrame({'selected_scenario': [True] * len(scenario_dict) ,'scenario_name': list(scenario_dict.keys())})
        values_dict = {
            f'{self.study_name}.{scatter_scenario}.samples_df': scenario_df
        }

        for scenario_name, scenario_specific_data in scenario_dict.items():
            scenarioUseCase = StudyIEANZE(execution_engine=self.execution_engine)
            scenarioUseCase.study_name = f'{self.study_name}.{scatter_scenario}.{scenario_name}'
            s_values_dict = scenarioUseCase.setup_usecase()

            for key, val in scenario_specific_data.items():
                full_varname = f'{self.study_name}.{scatter_scenario}.{scenario_name}.{key}'
                s_values_dict[full_varname] = val
            values_dict.update(s_values_dict)

        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
