'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2024/01/18 Copyright 2024 Capgemini

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

import numpy as np
import pandas as pd

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.dice.dice_model.usecase import (
    Study as dice_usecase,
)
from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.post_processing.post_processing_factory import (
    PostProcessingFactory,
)


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)
        self.data_dir = join(dirname(__file__), 'data')

    def setup_usecase(self, study_folder_path=None):
        dice_ms_usecase = dice_usecase(execution_engine=self.execution_engine)

        self.scatter_scenario = 'Control rate scenarios'
        # Set public values at a specific namespace
        dice_ms_usecase.study_name = f'{self.study_name}.{self.scatter_scenario}'
        setup_data_list = dice_ms_usecase.setup_usecase()

        # setup_data_list[0].update(public_values)
        year_start = 2015
        year_end = GlossaryCore.YearEndDefault
        time_step = 5
        nb_per = round(
            (year_end - year_start) / time_step + 1)
        years = np.arange(year_start, year_end + 1, time_step)

        # scenario A
        scenario_A = 'Base case scenario'
        rate_A = [0.03, 0.0323, 0.0349, 0.0377, 0.0408, 0.0441, 0.0476, 0.0515,
                  0.0556, 0.0601, 0.0650, 0.0702, 0.0759, 0.0821, 0.0887, 0.0959, 0.1036, 0.1120]
        control_rate_A = pd.DataFrame({'year': years, 'value': rate_A})

        # scenario C
        scenario_C = 'Nordhaus optimal scenario'
        rate_C = [0.039, 0.195300068397082, 0.218457488964647, 0.243203228631712, 0.2694976095, 0.297314850299999, 0.3266363445,
                  0.3574480742, 0.389738332999999, 0.4234959626, 0.458708928, 0.495363113, 0.5334412573, 0.5729219824, 0.6137788811, 0.655979684399999, 0.699485574799999, 0.744250816099999]
        control_rate_C = pd.DataFrame({'year': years, 'value': rate_C})

        # scenario B
        scenario_B = 'Zero emission from 2030 scenario'
        rate_B = [0.03, 0.0323, 0.5, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        control_rate_B = pd.DataFrame({'year': years, 'value': rate_B})

        # scenario D
        scenario_D = 'Zero emission from 2020 scenario'
        rate_D = [0.03, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        control_rate_D = pd.DataFrame({'year': years, 'value': rate_D})

        scenario_E = 'Zero emission from 2050 scenario'
        rate_E = [0.03, 0.05, 0.1, 0.15, 0.2, 0.30, 0.50, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        control_rate_E = pd.DataFrame({'year': years, 'value': rate_E})

        # MDA initialization values
        data = np.zeros(nb_per)
        economics_df = pd.DataFrame({'year': years,
                                     'saving_rate': data,
                                     GlossaryCore.GrossOutput: data,
                                     GlossaryCore.OutputNetOfDamage: data,
                                     GlossaryCore.NetOutput: data,
                                     GlossaryCore.PopulationValue: data,
                                     GlossaryCore.Productivity: data,
                                     GlossaryCore.ProductivityGrowthRate: data,
                                     GlossaryCore.Consumption: data,
                                     GlossaryCore.PerCapitaConsumption: data,
                                     GlossaryCore.Capital: data,
                                     GlossaryCore.InvestmentsValue: data,
                                     'interest_rate': data},
                                    index=np.arange(year_start, year_end + 1, time_step))

        values_dict = {}
        scenario_list = [scenario_A, scenario_C,
                         scenario_D, scenario_B, scenario_E]
        for scenario in scenario_list:
            values_dict[f'{self.study_name}.{self.scatter_scenario}.{scenario}.{GlossaryCore.EconomicsDfValue}'] = economics_df

        values_dict[f'{self.study_name}.{self.scatter_scenario}.samples_df'] = pd.DataFrame({
            'selected_scenario': [True for _ in scenario_list],
            'scenario_name': scenario_list})
        values_dict[f'{self.study_name}.{self.scatter_scenario}.{scenario_A}.emissions_control_rate'] = control_rate_A
        values_dict[f'{self.study_name}.{self.scatter_scenario}.{scenario_B}.emissions_control_rate'] = control_rate_B
        values_dict[f'{self.study_name}.{self.scatter_scenario}.{scenario_C}.emissions_control_rate'] = control_rate_C
        values_dict[f'{self.study_name}.{self.scatter_scenario}.{scenario_D}.emissions_control_rate'] = control_rate_D
        values_dict[f'{self.study_name}.{self.scatter_scenario}.{scenario_E}.emissions_control_rate'] = control_rate_E
        setup_data_list[0].update(values_dict)

        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()

    post_processing_factory = PostProcessingFactory()

    # all_post_processings = post_processing_factory.get_all_post_processings(
    # execution_engine=uc_cls.execution_engine, filters_only=False,
    # as_json=False, for_test=False)

    charts = post_processing_factory.get_post_processing_by_namespace(
        uc_cls.execution_engine, f'{uc_cls.study_name}.Post-processing', None)
