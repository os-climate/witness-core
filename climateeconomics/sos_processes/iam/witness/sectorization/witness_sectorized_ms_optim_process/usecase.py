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
from climateeconomics.sos_processes.iam.witness.sectorization.witness_sectorization_optim.usecase_witness_sectorization_optim import (
    Study as StudyMonoOptim,
)


class Study(ClimateEconomicsStudyManager):
    UC1 = "- Damage, - Tax"
    UC2 = "+ Damage, + Tax, Fossil only"
    UC3 = "+ Damage, + Tax, No CCUS"
    UC4 = "+ Damage, + Tax, All technos"

    def __init__(self, bspline=False, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')

    def setup_usecase(self, study_folder_path=None):

        scatter_scenario = 'optimization scenarios'

        scenario_dict = {
            self.UC1: StudyMonoOptim,
            self.UC2: StudyMonoOptim,
            self.UC3: StudyMonoOptim,
            self.UC4: StudyMonoOptim,
        }
        variables_to_deactivate_in_design_space = {
            self.UC1: [
                'clean_energy.CleanEnergySimpleTechno.clean_energy_CleanEnergySimpleTechno_array_mix',
                'clean_energy_CleanEnergySimpleTechno_utilization_ratio_array',
                'carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix',
                'carbon_capture.flue_gas_capture.FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix',
                'carbon_capture.direct_air_capture.DirectAirCaptureTechno_utilization_ratio_array',
                'carbon_capture.flue_gas_capture.FlueGasTechno_utilization_ratio_array',
                'carbon_storage.CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix',
                'carbon_storage.CarbonStorageTechno_utilization_ratio_array',
            ],
            self.UC2: [
                'clean_energy.CleanEnergySimpleTechno.clean_energy_CleanEnergySimpleTechno_array_mix',
                'clean_energy_CleanEnergySimpleTechno_utilization_ratio_array',
                'carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix',
                'carbon_capture.flue_gas_capture.FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix',
                'carbon_capture.direct_air_capture.DirectAirCaptureTechno_utilization_ratio_array',
                'carbon_capture.flue_gas_capture.FlueGasTechno_utilization_ratio_array',
                'carbon_storage.CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix',
                'carbon_storage.CarbonStorageTechno_utilization_ratio_array',
            ],
            self.UC3: [
                'carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix',
                'carbon_capture.flue_gas_capture.FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix',
                'carbon_capture.direct_air_capture.DirectAirCaptureTechno_utilization_ratio_array',
                'carbon_capture.flue_gas_capture.FlueGasTechno_utilization_ratio_array',
                'carbon_storage.CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix',
                'carbon_storage.CarbonStorageTechno_utilization_ratio_array',
            ],
            self.UC4: [],
        }

        scenario_df = pd.DataFrame({'selected_scenario': [True] * len(scenario_dict) ,'scenario_name': list(scenario_dict.keys())})
        values_dict = {
            f'{self.study_name}.{scatter_scenario}.samples_df': scenario_df,
            f'{self.study_name}.n_subcouplings_parallel': min(16, len(scenario_df.loc[scenario_df['selected_scenario']]))
        }

        for scenario_name, studyClass in scenario_dict.items():
            scenarioUseCase = studyClass(execution_engine=self.execution_engine)
            scenarioUseCase.study_name = f'{self.study_name}.{scatter_scenario}.{scenario_name}'
            scenarioData = scenarioUseCase.setup_usecase()

            dspace_scenario_varname = self.get_fullname_in_values_dict(values_dict=scenarioData, varname='design_space')[0]
            dspace = scenarioData[dspace_scenario_varname]
            dvar_to_deactivate_scenario = variables_to_deactivate_in_design_space[scenario_name]
            dspace.loc[dspace['variable'].isin(dvar_to_deactivate_scenario), 'enable_variable'] = False
            scenarioData[dspace_scenario_varname] = dspace

            values_dict.update(scenarioData)

        values_dict.update({f"{self.study_name}.{scatter_scenario}.{self.UC1}.WITNESS_MDO.WITNESS_Eval.WITNESS.co2_damage_price_percentage": 0.0})
        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
    uc_cls.test()
