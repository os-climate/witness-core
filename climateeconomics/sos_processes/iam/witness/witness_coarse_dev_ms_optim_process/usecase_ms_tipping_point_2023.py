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

from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_3_no_ccs_damage_high_tax import (
    Study as Study3,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_2023_nze_2050 import (
    Study as StudyNZE,
)


class Study(ClimateEconomicsStudyManager):
    TIPPING_POINT = 'Tipping point'
    TIPPING_POINT_LIST = [6, 4.5, 3.5]
    SEP = ' '
    UNIT = 'deg C'

    UC1 = "- Damage, - Tax"
    UC3_tp1 = "+ Damage, + Tax, No CCUS" + ', ' + TIPPING_POINT + SEP + str(TIPPING_POINT_LIST[0]).replace('.', '_') + UNIT
    UC3_tp2 = "+ Damage, + Tax, No CCUS" + ', ' + TIPPING_POINT + SEP + str(TIPPING_POINT_LIST[1]).replace('.',
                                                                                                           '_') + UNIT
    UC3_tp3 = "+ Damage, + Tax, No CCUS" + ', ' + TIPPING_POINT + SEP + str(TIPPING_POINT_LIST[2]).replace('.',
                                                                                                           '_') + UNIT
    UC4_tp1 = "+ Damage, + Tax, All technos NZE" + ', ' + TIPPING_POINT + SEP + str(TIPPING_POINT_LIST[0]).replace('.', '_') + UNIT
    UC4_tp2 = "+ Damage, + Tax, All technos NZE" + ', ' + TIPPING_POINT + SEP + str(TIPPING_POINT_LIST[1]).replace('.',
                                                                                                                   '_') + UNIT
    UC4_tp3 = "+ Damage, + Tax, All technos NZE" + ', ' + TIPPING_POINT + SEP + str(TIPPING_POINT_LIST[2]).replace('.',
                                                                                                                   '_') + UNIT

    def __init__(self, year_start=2023, bspline=False, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')
        self.year_start = year_start

    def setup_usecase(self, study_folder_path=None):

        scatter_scenario = 'optimization scenarios'

        scenario_dict = {
            self.UC1: Study1,
            self.UC3_tp1: Study3,
            self.UC3_tp2: Study3,
            self.UC3_tp3: Study3,
            self.UC4_tp1: StudyNZE,
            self.UC4_tp2: StudyNZE,
            self.UC4_tp3: StudyNZE,
        }

        # changing the tipping point

        scenario_df = pd.DataFrame({'selected_scenario': [True] * len(scenario_dict) ,'scenario_name': list(scenario_dict.keys())})
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

        values_dict.update({f"{self.study_name}.{scatter_scenario}.{scenario_name}.WITNESS_MDO.max_iter": 1 for scenario_name in scenario_dict.keys()})
        # update the tipping point value
        values_dict.update({
            f'{self.study_name}.{scatter_scenario}.{self.UC3_tp1}.WITNESS_MDO.WITNESS_Eval.WITNESS.Damage.tp_a3': self.TIPPING_POINT_LIST[0],
            f'{self.study_name}.{scatter_scenario}.{self.UC3_tp2}.WITNESS_MDO.WITNESS_Eval.WITNESS.Damage.tp_a3': self.TIPPING_POINT_LIST[1],
            f'{self.study_name}.{scatter_scenario}.{self.UC3_tp3}.WITNESS_MDO.WITNESS_Eval.WITNESS.Damage.tp_a3': self.TIPPING_POINT_LIST[2],
            f'{self.study_name}.{scatter_scenario}.{self.UC4_tp1}.WITNESS_MDO.WITNESS_Eval.WITNESS.Damage.tp_a3': self.TIPPING_POINT_LIST[0],
            f'{self.study_name}.{scatter_scenario}.{self.UC4_tp2}.WITNESS_MDO.WITNESS_Eval.WITNESS.Damage.tp_a3': self.TIPPING_POINT_LIST[1],
            f'{self.study_name}.{scatter_scenario}.{self.UC4_tp3}.WITNESS_MDO.WITNESS_Eval.WITNESS.Damage.tp_a3': self.TIPPING_POINT_LIST[2],
        })

        values_dict = self.update_dataframes_with_year_star(values_dict=values_dict, year_start=self.year_start)

        return values_dict

if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
