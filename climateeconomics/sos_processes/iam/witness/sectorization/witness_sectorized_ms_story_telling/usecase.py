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
from climateeconomics.database.story_telling.story_telling_db import StoryTellingDatabase
from climateeconomics.glossarycore import GlossaryCore

from climateeconomics.sos_processes.iam.witness.sectorization.witness_sectorization.usecase_witness_coarse_sectorization import Study as StudyRef


class Study(ClimateEconomicsStudyManager):
    UC1 = "- Damage, - Tax"
    UC2 = "+ Damage, + Tax, Fossil only"
    UC3 = "+ Damage, + Tax, No CCUS"
    UC4 = "+ Damage, + Tax, CCUS"

    def __init__(self, bspline=True, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = GlossaryCore.YearEndDefault

    def setup_usecase(self, study_folder_path=None):

        scatter_scenario = 'scenarios'

        scenario_dict = {
            self.UC1: StudyRef,
            self.UC2: StudyRef,
            self.UC3: StudyRef,
            self.UC4: StudyRef,
        }

        scenario_df = pd.DataFrame(
            {'selected_scenario': [True] * len(scenario_dict), 'scenario_name': list(scenario_dict.keys())})
        values_dict = {
            f'{self.study_name}.{scatter_scenario}.samples_df': scenario_df,
            f'{self.study_name}.inner_mda_name': "MDAGaussSeidel",
            f'{self.study_name}.max_mda_iter': 100,
            f'{self.study_name}.n_subcouplings_parallel': min(16, len(scenario_df.loc[scenario_df['selected_scenario']]))
        }
        scenario_datas = {
            self.UC1: {
                GlossaryCore.invest_mix: StoryTellingDatabase.FullFossilEnergyInvestMix.get_all_cols_between_years(self.year_start, self.year_end),
                f"{GlossaryCore.EnergyMix}.{GlossaryCore.ShareSectorInvestmentDfValue}": StoryTellingDatabase.FullFossilShareEnergyInvest.get_all_cols_between_years(self.year_start, self.year_end),
                f"{GlossaryCore.CCUS}.{GlossaryCore.ShareSectorInvestmentDfValue}": StoryTellingDatabase.NoCCUSInvestShare.get_all_cols_between_years(self.year_start, self.year_end),
            },
            self.UC2: {
                GlossaryCore.invest_mix: StoryTellingDatabase.FullFossilEnergyInvestMix.get_all_cols_between_years(self.year_start, self.year_end),
                f"{GlossaryCore.EnergyMix}.{GlossaryCore.ShareSectorInvestmentDfValue}": StoryTellingDatabase.FullFossilShareEnergyInvest.get_all_cols_between_years(self.year_start, self.year_end),
                f"{GlossaryCore.CCUS}.{GlossaryCore.ShareSectorInvestmentDfValue}": StoryTellingDatabase.NoCCUSInvestShare.get_all_cols_between_years(self.year_start, self.year_end),
            },
            self.UC3: {
                GlossaryCore.invest_mix: StoryTellingDatabase.BAUEnergyInvestMix.get_all_cols_between_years(self.year_start, self.year_end),
                f"{GlossaryCore.EnergyMix}.{GlossaryCore.ShareSectorInvestmentDfValue}": StoryTellingDatabase.BusineesAsUsualShareEnergyInvest.get_all_cols_between_years(self.year_start, self.year_end),
                f"{GlossaryCore.CCUS}.{GlossaryCore.ShareSectorInvestmentDfValue}": StoryTellingDatabase.NoCCUSInvestShare.get_all_cols_between_years(self.year_start, self.year_end),
            },
            self.UC4: {
                GlossaryCore.invest_mix: StoryTellingDatabase.NZEEnergyInvestMix.get_all_cols_between_years(self.year_start, self.year_end),
                f"{GlossaryCore.EnergyMix}.{GlossaryCore.ShareSectorInvestmentDfValue}": StoryTellingDatabase.NZEShareEnergyInvest.get_all_cols_between_years(self.year_start, self.year_end),
                f"{GlossaryCore.CCUS}.{GlossaryCore.ShareSectorInvestmentDfValue}": StoryTellingDatabase.NZEShareCCUSInvest.get_all_cols_between_years(self.year_start, self.year_end),
            },
        }
        for scenario_name, study_class in scenario_dict.items():
            study_instance = study_class()
            study_instance.study_name = f"{self.study_name}.{scatter_scenario}.{scenario_name}"
            study_data = study_instance.setup_usecase()
            study_data.update({
                f"{study_instance.study_name}.{k}": v for k,v in scenario_datas[scenario_name].items()
            })
            values_dict.update(study_data)


        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
