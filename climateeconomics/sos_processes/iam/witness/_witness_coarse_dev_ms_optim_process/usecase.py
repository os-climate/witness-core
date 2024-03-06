'''
Copyright 2022 Airbus SAS
Modifications on 2023/03/27-2023/11/03 Copyright 2023 Capgemini

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
from os.path import join, dirname

import pandas as pd

from climateeconomics.core.tools.ClimateEconomicsStudyManager import ClimateEconomicsStudyManager
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_2_witness_coarse_gdp_model_wo_damage_wo_co2_tax import \
    Study as Study2
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_3_witness_coarse_gdp_model_wo_damage_w_co2_tax import \
    Study as Study3
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_4_witness_coarse_gdp_model_wo_damage_wo_co2_tax_ccs2020_ren2020 import \
    Study as Study4
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_5_witness_coarse_gdp_model_wo_damage_wo_co2_tax_ccs2020 import \
    Study as Study5
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_6_witness_coarse_gdp_model_w_damage_wo_co2_tax import \
    Study as Study6
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_7_witness_coarse_gdp_model_w_damage_w_co2_tax import \
    Study as Study7


class Study(ClimateEconomicsStudyManager):

    def __init__(self, bspline=False, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')
        self.scatter_scenario = 'mda_ms_story_telling'

    def setup_usecase(self, study_folder_path=None):

        scenario_dict = {
            "usecase_2": Study2(execution_engine=self.execution_engine),
            "usecase_3": Study3(execution_engine=self.execution_engine),
            "usecase_4": Study4(execution_engine=self.execution_engine),
            "usecase_5": Study5(execution_engine=self.execution_engine),
            "usecase_6": Study6(execution_engine=self.execution_engine),
            "usecase_7": Study7(execution_engine=self.execution_engine)
        }



        scenario_list = list(scenario_dict.keys())
        values_dict = {}

        scenario_df = pd.DataFrame({'selected_scenario': [True] * len(scenario_list),
                                    'scenario_name': scenario_list})
        values_dict[f'{self.study_name}.{self.scatter_scenario}.samples_df'] = scenario_df
        values_dict[f'{self.study_name}.{self.scatter_scenario}.scenario_list'] = scenario_list

        # assumes max of 16 cores per computational node
        values_dict[f'{self.study_name}.n_subcouplings_parallel'] = min(16, len(
            scenario_df.loc[scenario_df['selected_scenario'] == True]))
        # setup each scenario (mda settings ignored)
        for scenario, uc in scenario_dict.items():
            uc.study_name = f'{self.study_name}.{self.scatter_scenario}.{scenario}'
            for dict_data in uc.setup_usecase():
                values_dict.update(dict_data)

        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
