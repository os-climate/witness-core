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
from copy import copy
import numpy as np
from climateeconomics.core.tools.ClimateEconomicsStudyManager import ClimateEconomicsStudyManager
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_witness_optim_invest_distrib import \
    Study as usecase_witness


class Study(ClimateEconomicsStudyManager):

    def __init__(self, run_usecase=False, execution_engine=None, year_start=2020, year_end=2100, time_step=1):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step

    def setup_usecase(self):
        witness_uc = usecase_witness()
        witness_uc.study_name = self.study_name
        data_witness = witness_uc.setup_usecase()
        
        dspace = witness_uc.witness_uc.dspace

        # deactivate CCS and Renewable design variables and set values to their 2020 value.
        list_design_var_to_clean = ['red_meat_calories_per_day_ctrl', 'white_meat_calories_per_day_ctrl', 'vegetables_and_carbs_calories_per_day_ctrl', 'milk_and_eggs_calories_per_day_ctrl', 'forest_investment_array_mix', 'deforestation_investment_ctrl']
        var_to_set_to_2020_level = ['renewable.RenewableSimpleTechno.renewable_RenewableSimpleTechno_array_mix',
                                    'carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix',
                                    'carbon_capture.flue_gas_capture.FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix',
                                    'carbon_storage.CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix',
                                    'carbon_capture_direct_air_capture.DirectAirCaptureTechno_utilization_ratio_array',
                                    'carbon_capture_flue_gas_capture.FlueGasTechno_utilization_ratio_array',]
        serie_index = dspace['variable'].isin(var_to_set_to_2020_level)
        activated_elem_column = copy(dspace['activated_elem'])
        value_column = copy(dspace['value'])
        enable_variable_column = copy(dspace['enable_variable'])
        for index, value in serie_index.items():
            if value:
                activated_elem_column[index] = [True for value in dspace['activated_elem'][index]]
                enable_variable_column[index] = False
                value_column[index] = np.array([dspace['value'][index][0] for val in dspace['value'][index]])

        dspace['activated_elem'] = activated_elem_column
        dspace['value'] = value_column
        dspace['enable_variable'] = enable_variable_column

        # clean dspace
        dspace.drop(dspace.loc[dspace['variable'].isin(list_design_var_to_clean)].index, inplace=True)

        # clean dspace descriptor 
        dvar_descriptor = witness_uc.witness_uc.design_var_descriptor
        
        updated_dvar_descriptor = {k:v for k,v in dvar_descriptor.items() if k not in list_design_var_to_clean}
        updated_data = {f'{self.study_name}.{witness_uc.optim_name}.{witness_uc.coupling_name}.{witness_uc.extra_name}.assumptions_dict': {'compute_gdp': True,
                                                                'compute_climate_impact_on_gdp': False,
                                                                'activate_climate_effect_population': False,
                                                                'invest_co2_tax_in_renewables': False
                                                               },
                        f'{self.study_name}.{witness_uc.optim_name}.design_space' : dspace,
                        f'{self.study_name}.{witness_uc.optim_name}.{witness_uc.coupling_name}.{witness_uc.witness_uc.designvariable_name}.design_var_descriptor': updated_dvar_descriptor}
        data_witness.append(updated_data)

        data_witness.append({
            f"{self.study_name}.{witness_uc.optim_name}.{witness_uc.coupling_name}.{witness_uc.extra_name}.ccs_price_percentage": 0.0,
            f"{self.study_name}.{witness_uc.optim_name}.{witness_uc.coupling_name}.{witness_uc.extra_name}.co2_damage_price_percentage": 0.0,
        })

        return data_witness


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.test()
