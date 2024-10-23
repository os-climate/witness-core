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

from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_witness_optim_invest_distrib import (
    Study as StudyOptimInvestDistrib,
)
from tools.design_space_creator import (
    get_ine_dvar_descr,
    make_dspace_Ine,
    make_dspace_invests,
    make_dspace_utilization_ratio,
)


class Study(StudyOptimInvestDistrib):
    def __init__(self, run_usecase=False, execution_engine=None, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault):
        super().__init__(year_start=year_start,
                         year_end=year_end,
                         file_path=__file__,
                         run_usecase=run_usecase,
                         execution_engine=execution_engine)

        self.test_post_procs = False

    def setup_usecase(self, study_folder_path=None):
        self.test_post_procs = False
        data_witness = super().setup_usecase()

        # update fossil invest & utilization ratio lower bound to not be too low
        min_invest = 1.
        max_invest = 8000.
        dspace_invests = {
            'fossil.FossilSimpleTechno.fossil_FossilSimpleTechno_array_mix': [300., 300., max_invest, True],
            f"{GlossaryCore.clean_energy}.{GlossaryCore.CleanEnergySimpleTechno}.{GlossaryCore.clean_energy}_{GlossaryCore.CleanEnergySimpleTechno}_array_mix": [min_invest, min_invest, max_invest, True],
            'carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix': [min_invest, min_invest, max_invest, False],
            'carbon_capture.flue_gas_capture.FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix': [min_invest, min_invest, max_invest, False],
            'carbon_storage.CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix': [min_invest, min_invest, max_invest, False],
        }
        dspace_invests = make_dspace_invests(dspace_invests, self.year_start)
        min_UR = 50.
        dspace_UR = {
            'fossil_FossilSimpleTechno_utilization_ratio_array': [min_UR, min_UR, 100., True],
            f"{GlossaryCore.clean_energy}_{GlossaryCore.CleanEnergySimpleTechno}_utilization_ratio_array": [min_UR, min_UR, 100., True],
            'carbon_capture.direct_air_capture.DirectAirCaptureTechno_utilization_ratio_array': [min_UR, min_UR, 100., False],
            'carbon_capture.flue_gas_capture.FlueGasTechno_utilization_ratio_array': [min_UR, min_UR, 100., False],
            'carbon_storage.CarbonStorageTechno_utilization_ratio_array': [min_UR, min_UR, 100., False],

        }
        dspace_UR = make_dspace_utilization_ratio(dspace_UR)

        # dspace pour Ine
        dspace_Ine = make_dspace_Ine(enable_variable=False)
        dspace = pd.concat([dspace_invests, dspace_UR, dspace_Ine])

        # update design var descriptor with Ine variable
        dvar_descriptor = data_witness[f'{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.DesignVariables.design_var_descriptor']
        design_var_descriptor_ine_variable = get_ine_dvar_descr(self.year_start, self.year_end)

        dvar_descriptor.update({
            "share_non_energy_invest_ctrl": design_var_descriptor_ine_variable
        })

        # Deactivate damage
        updated_data = {
            f'{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.assumptions_dict': {
                'compute_gdp': True,
                'compute_climate_impact_on_gdp': False,
                'activate_climate_effect_population': False,
                'activate_pandemic_effects': False
            },
            f'{self.study_name}.{self.optim_name}.design_space': dspace,
        }
        data_witness.update(updated_data)

        # Put low tax
        data_witness.update({
            f"{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.ccs_price_percentage": 0.0,
            f"{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.co2_damage_price_percentage": 0.0,
            f"{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.share_non_energy_invest_ctrl": np.array([DatabaseWitnessCore.ShareInvestNonEnergy.value] * (GlossaryCore.NB_POLES_COARSE - 1)),
        })

        return data_witness


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()

