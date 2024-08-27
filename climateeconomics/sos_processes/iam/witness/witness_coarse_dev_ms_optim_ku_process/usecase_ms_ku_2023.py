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
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_ms_optim_ku_process.usecase_ms_ku import (
    Study as Study2020,
)


class Study(Study2020):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(filename=__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.test_post_procs = True

    def setup_usecase(self, study_folder_path=None):
        values_dict = super().setup_usecase(study_folder_path)
        values_dict.update(
            {f"{self.study_name}.{self.scatter_scenario}.{scenario_name}.{GlossaryCore.YearStart}": 2023 for
             scenario_name
             in self.scenario_dict.keys()})

        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
