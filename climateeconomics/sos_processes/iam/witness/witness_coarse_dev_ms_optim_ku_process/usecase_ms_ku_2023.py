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
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_ms_optim_ku_process.usecase_ms_ku import (
    Study as Study2020,
)


class Study(Study2020):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(year_start=2023, filename=__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.test_post_procs = True

    def setup_usecase(self, study_folder_path=None):
        values_dict = super().setup_usecase(study_folder_path)
        year_start_varnames = list(filter(lambda x: f".{GlossaryCore.YearStart}" in x, values_dict.keys()))
        values_dict.update({varname: 2023 for varname in year_start_varnames})

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
