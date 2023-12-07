'''
Copyright 2022 Airbus SAS
Modifications on 27/11/2023 Copyright 2023 Capgemini

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

import numpy as np
import pandas as pd

from climateeconomics.core.tools.ClimateEconomicsStudyManager import ClimateEconomicsStudyManager
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_1_witness_coarse_mda_fixed_gdp_wo_damage_wo_co2_tax import \
    Study as usecase1
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_2_witness_coarse_mda_gdp_model_wo_damage_wo_co2_tax import \
    Study as usecase2
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_3_witness_coarse_mda_gdp_model_wo_damage_w_co2_tax import \
    Study as usecase3
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_4_witness_coarse_mda_gdp_model_w_damage_wo_co2_tax import \
    Study as usecase4
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_5_witness_coarse_mda_gdp_model_w_damage_wo_co2_tax import \
    Study as usecase5
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_6_witness_coarse_mda_gdp_model_w_damage_wo_co2_tax import \
    Study as usecase6
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_7_witness_coarse_mda_gdp_model_w_damage_w_co2_tax import \
    Study as usecase7
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory


class Study(ClimateEconomicsStudyManager):

    def __init__(self, bspline=False, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')

    def setup_usecase(self):
        uc1 = usecase1(execution_engine=self.execution_engine)
        uc2 = usecase2(execution_engine=self.execution_engine)

        self.scatter_scenario = 'mda_scenarios'

        values_dict = {}
        scenario_list = ['usecase_1', 'usecase_2']
        scenario_df = pd.DataFrame({'selected_scenario': [True, True],
                                    'scenario_name': scenario_list})
        values_dict[f'{self.study_name}.{self.scatter_scenario}.scenario_df'] = scenario_df
        values_dict[f'{self.study_name}.{self.scatter_scenario}.scenario_list'] = scenario_list
        values_dict[f'{self.study_name}.{self.scatter_scenario}.builder_mode'] = 'multi_instance'
        #values_dict[f'{self.study_name}.epsilon0'] = 1.0
        #values_dict[f'{self.study_name}.n_subcouplings_parallel'] = 2

        for scenario, uc in {scenario_list[0]: uc1,
                             scenario_list[1]: uc2}.items():
            uc.study_name = f'{self.study_name}.{self.scatter_scenario}.{scenario}'
            for dict_data in uc.setup_usecase():
                values_dict.update(dict_data)



        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
    post_processing_factory = PostProcessingFactory()
    post_processing_factory.get_post_processing_by_namespace(
        uc_cls.execution_engine, f'{uc_cls.study_name}.Post-processing', [])
    all_post_processings = post_processing_factory.get_all_post_processings(
         uc_cls.execution_engine, False, as_json=False, for_test=False)

    for namespace, post_proc_list in all_post_processings.items():
        for chart in post_proc_list:
            chart.to_plotly().show()

    print()
