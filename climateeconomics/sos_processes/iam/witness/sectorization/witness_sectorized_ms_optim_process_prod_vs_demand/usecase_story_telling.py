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

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.database.story_telling.story_telling_db import StDB
from climateeconomics.sos_processes.iam.witness.sectorization.witness_sectorization_optim_prod_vs_demand.usecase_prod_vs_demand import (
    Study as StudyMonoOptim,
)


class Study(ClimateEconomicsStudyManager):

    def __init__(self, bspline=True, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')

    def setup_usecase(self, study_folder_path=None):

        scatter_scenario = 'scenarios'

        scenario_dict = {
            StDB.USECASE2: {
                "invest_mix": StDB.FullFossilEnergyInvestMix.value,
                "assumptions_dict": ClimateEcoDiscipline.assumptions_dict_no_damages,
            },
            StDB.USECASE2B: {
                "invest_mix": StDB.FullFossilEnergyInvestMix.value,
            },
            StDB.USECASE4: {
                "invest_mix": StDB.UC4EnergyInvestMix.value,
            },
            StDB.USECASE7: {
                "invest_mix": StDB.UC7EnergyInvestMix.value,
            },

        }


        scenario_df = pd.DataFrame({'selected_scenario': [True] * len(scenario_dict) ,'scenario_name': list(scenario_dict.keys())})
        values_dict = {
            f'{self.study_name}.{scatter_scenario}.samples_df': scenario_df,
            f'{self.study_name}.n_subcouplings_parallel': min(16, len(scenario_df.loc[scenario_df['selected_scenario']]))
        }

        for scenario_name, scenario_specific_data in scenario_dict.items():
            scenarioUseCase = StudyMonoOptim(execution_engine=self.execution_engine)
            scenarioUseCase.study_name = f'{self.study_name}.{scatter_scenario}.{scenario_name}'
            s_values_dict = scenarioUseCase.setup_usecase()

            for key, val in scenario_specific_data.items():
                full_varname = self.get_fullname_in_values_dict(values_dict=s_values_dict, varname=key)[0]
                s_values_dict[full_varname] = val
            values_dict.update(s_values_dict)

        values_dict.update({f"{self.study_name}.{scatter_scenario}.{StDB.USECASE2}.MDO.WITNESS_Eval.WITNESS.co2_damage_price_percentage": 0.0})

        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run(for_test=True)
