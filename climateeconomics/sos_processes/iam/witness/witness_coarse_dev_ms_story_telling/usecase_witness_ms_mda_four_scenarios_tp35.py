'''
Copyright 2023 Capgemini

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
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev.usecase_witness_coarse_new import (
    Study as usecase_witness_mda,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_ms_story_telling.usecase_witness_ms_mda import (
    Study as usecase_ms_mda,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_2_witness_coarse_mda_gdp_model_wo_damage_wo_co2_tax import (
    Study as usecase2,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_2b_witness_coarse_mda_gdp_model_w_damage_wo_co2_tax import (
    Study as usecase2b,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_4_witness_coarse_mda_gdp_model_w_damage_wo_co2_tax import (
    Study as usecase4,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_7_witness_coarse_mda_gdp_model_w_damage_w_co2_tax import (
    Study as usecase7,
)


class Study(ClimateEconomicsStudyManager):

    USECASE2 = '- damage - tax, fossil 100%'
    USECASE2B ='+ damage - tax, fossil 100%'
    USECASE4 = '+ damage - tax, fossil 40%'
    USECASE7 = '+ damage + tax, NZE'

    def __init__(self, file_path=__file__, bspline=False, run_usecase=False, execution_engine=None, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault):
        super().__init__(file_path=file_path, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')
        self.check_outputs = False
        self.test_post_procs = False
        self.year_start = year_start
        self.year_end = year_end

    def setup_usecase(self, study_folder_path=None):

        self.scatter_scenario = 'mda_scenarios'

        scenario_dict = {usecase_ms_mda.USECASE2: usecase2(execution_engine=self.execution_engine, year_start=self.year_start, year_end=self.year_end),
                         usecase_ms_mda.USECASE2B: usecase2b(execution_engine=self.execution_engine, year_start=self.year_start, year_end=self.year_end),
                         usecase_ms_mda.USECASE4: usecase4(execution_engine=self.execution_engine, year_start=self.year_start, year_end=self.year_end),
                         usecase_ms_mda.USECASE7: usecase7(execution_engine=self.execution_engine, year_start=self.year_start, year_end=self.year_end),
                         }

        scenario_list = list(scenario_dict.keys())
        values_dict = {}

        scenario_df = pd.DataFrame({'selected_scenario': [True] * len(scenario_list),
                                    'scenario_name': scenario_list})
        values_dict[f'{self.study_name}.{self.scatter_scenario}.samples_df'] = scenario_df
        values_dict[f'{self.study_name}.{self.scatter_scenario}.scenario_list'] = scenario_list
        # setup mda
        uc_mda = usecase_witness_mda(execution_engine=self.execution_engine, year_start=self.year_start, year_end=self.year_end)
        uc_mda.study_name = self.study_name  # mda settings on root coupling
        values_dict.update(uc_mda.setup_mda())
        # assumes max of 16 cores per computational node
        values_dict[f'{self.study_name}.n_subcouplings_parallel'] = min(16, len(scenario_df.loc[scenario_df['selected_scenario']]))
        # setup each scenario (mda settings ignored)
        for scenario, uc in scenario_dict.items():
            uc.study_name = f'{self.study_name}.{self.scatter_scenario}.{scenario}'
            for dict_data in uc.setup_usecase():
                values_dict.update(dict_data)

        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
