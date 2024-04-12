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

from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_witness_optim_invest_distrib import \
    Study as StudyOptimInvestDistrib


class Study(StudyOptimInvestDistrib):
    def __init__(self, run_usecase=False, execution_engine=None, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, time_step=1):
        super().__init__(year_start=year_start,
                         year_end=year_end,
                         time_step=time_step,
                         file_path=__file__,
                         run_usecase=run_usecase,
                         execution_engine=execution_engine)

    def setup_usecase(self, study_folder_path=None):
        
        data_witness = super().setup_usecase()
        
        dspace = data_witness[f'{self.study_name}.{self.optim_name}.design_space']


        # Deactivate damage
        updated_data = {
            f'{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.assumptions_dict': {
                'compute_gdp': True,
                'compute_climate_impact_on_gdp': False,
                'activate_climate_effect_population': False,
                'invest_co2_tax_in_renewables': False
            },
            f'{self.study_name}.{self.optim_name}.design_space': dspace,
        }
        data_witness.update(updated_data)

        # Let only fossil design vars
        var_to_deactive_and_set_to_lower_bound_value = [
            'renewable.RenewableSimpleTechno.renewable_RenewableSimpleTechno_array_mix',
            'carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix',
            'carbon_capture.flue_gas_capture.FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix',
            'carbon_storage.CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix',
            'renewable_RenewableSimpleTechno_utilization_ratio_array',
            'carbon_capture.direct_air_capture.DirectAirCaptureTechno_utilization_ratio_array',
            'carbon_capture.flue_gas_capture.FlueGasTechno_utilization_ratio_array',
            'carbon_storage.CarbonStorageTechno_utilization_ratio_array'
        ]

        serie_index = dspace['variable'].isin(var_to_deactive_and_set_to_lower_bound_value)

        for index_row, var_has_to_be_deactivate in serie_index.items():
            if var_has_to_be_deactivate:
                dspace.iloc[index_row]['value'] = dspace.iloc[index_row]['lower_bnd']

        dspace.loc[dspace['variable'].isin(var_to_deactive_and_set_to_lower_bound_value), 'enable_variable'] = False

        # Put low tax
        data_witness.update({
            f"{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.ccs_price_percentage": 25.0,
            f"{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.co2_damage_price_percentage": 25.0,
        })

        return data_witness


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()