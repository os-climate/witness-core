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
import pandas as pd

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_ms_story_telling.usecase_witness_ms_mda_four_scenarios_tp35 import (
    Study as StudyMSmdaTippingPoint35,
)


class Study(StudyMSmdaTippingPoint35):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(file_path=__file__, run_usecase=run_usecase, execution_engine=execution_engine, year_start=2023)
        self.test_post_procs = True

    def setup_usecase(self, study_folder_path=None):

        values_dict = super().setup_usecase()

        varnames = self.get_fullname_in_values_dict(values_dict, GlossaryCore.YearStart)
        for varname in varnames:
            values_dict[varname] = 2023
        tipping_point_variable = 'Damage.tp_a3'
        values_dict.update({
            f'{self.study_name}.{self.scatter_scenario}.{self.USECASE2}.{tipping_point_variable}': 6.081,
            f'{self.study_name}.{self.scatter_scenario}.{self.USECASE2B}.{tipping_point_variable}': 6.081,
            f'{self.study_name}.{self.scatter_scenario}.{self.USECASE4}.{tipping_point_variable}': 6.081,
            f'{self.study_name}.{self.scatter_scenario}.{self.USECASE7}.{tipping_point_variable}': 6.081,
        })

        values_dict_2023 = {}
        for key,value in values_dict.items():
            if isinstance(value, pd.DataFrame) and GlossaryCore.Years in value.columns:
                new_value = value.loc[value[GlossaryCore.Years] >= 2023]
                values_dict_2023[key] = new_value

        values_dict.update(values_dict_2023)

        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()