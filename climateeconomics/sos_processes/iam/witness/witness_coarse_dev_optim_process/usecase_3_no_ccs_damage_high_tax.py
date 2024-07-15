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

import numpy as np
import pandas as pd

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_witness_optim_invest_distrib import (
    Study as StudyOptimInvestDistrib,
)


class Study(StudyOptimInvestDistrib):
    def __init__(self, run_usecase=False, execution_engine=None, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, time_step=1):
        super().__init__(year_start=year_start,
                         year_end=year_end,
                         time_step=time_step,
                         file_path=__file__,
                         run_usecase=run_usecase,
                         execution_engine=execution_engine)
        self.test_post_procs = False

    def setup_usecase(self, study_folder_path=None):
        
        data_witness = super().setup_usecase()
        # update fossil invest & utilization ratio lower bound to not be too low
        min_invest = 1.
        max_invest = 3000.
        dspace_invests = {
            'fossil.FossilSimpleTechno.fossil_FossilSimpleTechno_array_mix': [10., 10., 5000., True],
            'renewable.RenewableSimpleTechno.renewable_RenewableSimpleTechno_array_mix': [300., 300., max_invest, True],
            'carbon_capture.direct_air_capture.{GlossaryEnergy.DirectAirCaptureTechno}.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix': [min_invest, min_invest, max_invest, False],
            'carbon_capture.flue_gas_capture.FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix': [min_invest, min_invest, max_invest, False],
            'carbon_storage.CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix': [min_invest, min_invest, max_invest, False],
        }
        dspace_invests = self.make_dspace_invests(dspace_invests)
        min_UR = 50.
        dspace_UR = {
            'fossil_FossilSimpleTechno_utilization_ratio_array': [min_UR, min_UR, 100., True],
            'renewable_RenewableSimpleTechno_utilization_ratio_array': [min_UR, min_UR, 100., True],
            'carbon_capture.direct_air_capture.DirectAirCaptureTechno_utilization_ratio_array': [min_UR, min_UR, 100., False],
            'carbon_capture.flue_gas_capture.FlueGasTechno_utilization_ratio_array': [min_UR, min_UR, 100., False],
            'carbon_storage.CarbonStorageTechno_utilization_ratio_array': [min_UR, min_UR, 100., False],
        }
        dspace_UR = self.make_dspace_utilization_ratio(dspace_UR)
        # dspace pour Ine
        dspace_Ine = self.make_dspace_Ine()
        dspace = pd.concat([dspace_invests, dspace_UR, dspace_Ine])

        # update design var descriptor with Ine variable
        dvar_descriptor = data_witness[f'{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.DesignVariables.design_var_descriptor']
        design_var_descriptor_ine_variable = self.get_ine_dvar_descr()
        
        dvar_descriptor.update({
            "share_non_energy_invest_ctrl": design_var_descriptor_ine_variable
        })

        # Activate damage
        updated_data = {
            f'{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.assumptions_dict': {
                'compute_gdp': True,
                'compute_climate_impact_on_gdp': True,
                'activate_climate_effect_population': True,
                'invest_co2_tax_in_renewables': False,
                'activate_pandemic_effects': False
            },
            f'{self.study_name}.{self.optim_name}.design_space': dspace,
        }

        data_witness.update(updated_data)


        # Put high tax
        data_witness.update({
            f"{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.ccs_price_percentage": 100.0,
            f"{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.co2_damage_price_percentage": 100.0,
            f"{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.share_non_energy_invest_ctrl": np.array([27.0] * (GlossaryCore.NB_POLES_COARSE - 1)),
        })

        return data_witness


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
