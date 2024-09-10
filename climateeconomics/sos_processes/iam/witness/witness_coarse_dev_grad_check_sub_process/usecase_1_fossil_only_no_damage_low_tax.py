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
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_grad_check_sub_process.usecase_witness_grad_check_optim_sub import (
    Study as StudySubOptim,
)


class Study(StudySubOptim):
    def __init__(self, run_usecase=False, execution_engine=None, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, time_step=1):
        super().__init__(year_start=year_start,
                         year_end=year_end,
                         time_step=time_step,
                         run_usecase=run_usecase,
                         execution_engine=execution_engine)

    def setup_usecase(self, study_folder_path=None):

        data_witness = super().setup_usecase()

        self.study_name += '_no_damage_low_tax'

        # Deactivate damage
        updated_data = {
            f'{self.study_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.assumptions_dict': {
                'compute_gdp': True,
                'compute_climate_impact_on_gdp': False,
                'activate_climate_effect_population': False,
                'activate_pandemic_effects': False
            },
        }
        data_witness[0].update(updated_data)

        # Put low tax
        data_witness[0].update({
            f"{self.study_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.ccs_price_percentage": 25.0,
            f"{self.study_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.co2_damage_price_percentage": 25.0,
        })

        return data_witness


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.test()
