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
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_ms_story_telling.usecase_witness_ms_mda_four_scenarios_tipping_points import (
    Study as uc_ms_mda,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_story_telling_optim_process.usecase_2_optim_story_telling import (
    Study as Study1,
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

    def setup_usecase(self, study_folder_path=None):

        scatter_scenario = 'optimization scenarios'

        #scenarios name:
        uc2 = uc_ms_mda.USECASE2
        uc4_tp1 = uc_ms_mda.USECASE4_TP1
        uc4_tp2 = uc_ms_mda.USECASE4_TP2
        uc4_tp3 = uc_ms_mda.USECASE4_TP3
        uc7_tp1 = uc_ms_mda.USECASE7_TP1
        uc7_tp2 = uc_ms_mda.USECASE7_TP2
        uc7_tp3 = uc_ms_mda.USECASE7_TP3

        scenario_dict = {
            uc2: Study1,
            uc4_tp1: Study3,
            uc4_tp2: Study3,
            uc4_tp3: Study3,
            uc7_tp1: Study4,
            uc7_tp2: Study4,
            uc7_tp3: Study4,
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
            for data in scenarioData:
                scenarioDatadict.update(data)
            values_dict.update(scenarioDatadict)

        values_dict.update({f"{self.study_name}.{scatter_scenario}.{scenario_name}.WITNESS_MDO.max_iter": 100 for scenario_name in scenario_dict.keys()})
        values_dict.update(
            {f"{self.study_name}.{scatter_scenario}.{scenario_name}.WITNESS_MDO.WITNESS_Eval.sub_mda_class": "MDAGaussSeidel" for scenario_name in
             scenario_dict.keys()})
        # update the tipping point value
        values_dict.update({
            f'{self.study_name}.{scatter_scenario}.{uc4_tp1}.WITNESS_MDO.WITNESS_Eval.WITNESS.Damage.tp_a3':uc_ms_mda.TIPPING_POINT_LIST[0],
            f'{self.study_name}.{scatter_scenario}.{uc4_tp2}.WITNESS_MDO.WITNESS_Eval.WITNESS.Damage.tp_a3':uc_ms_mda.TIPPING_POINT_LIST[1],
            f'{self.study_name}.{scatter_scenario}.{uc4_tp3}.WITNESS_MDO.WITNESS_Eval.WITNESS.Damage.tp_a3':uc_ms_mda.TIPPING_POINT_LIST[2],
            f'{self.study_name}.{scatter_scenario}.{uc7_tp1}.WITNESS_MDO.WITNESS_Eval.WITNESS.Damage.tp_a3':uc_ms_mda.TIPPING_POINT_LIST[0],
            f'{self.study_name}.{scatter_scenario}.{uc7_tp2}.WITNESS_MDO.WITNESS_Eval.WITNESS.Damage.tp_a3':uc_ms_mda.TIPPING_POINT_LIST[1],
            f'{self.study_name}.{scatter_scenario}.{uc7_tp3}.WITNESS_MDO.WITNESS_Eval.WITNESS.Damage.tp_a3':uc_ms_mda.TIPPING_POINT_LIST[2],
        })

        values_dict = self.update_dataframes_with_year_star(values_dict=values_dict, year_start=self.year_start)

        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()