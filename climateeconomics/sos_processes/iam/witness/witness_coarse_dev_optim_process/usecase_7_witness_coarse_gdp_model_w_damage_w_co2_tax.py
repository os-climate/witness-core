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

from climateeconomics.core.tools.ClimateEconomicsStudyManager import ClimateEconomicsStudyManager
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_witness_optim_invest_distrib import \
    Study as usecase_witness


class Study(ClimateEconomicsStudyManager):

    def __init__(self, run_usecase=False, execution_engine=None, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, time_step=1):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step

    def setup_usecase(self, study_folder_path=None):
        witness_uc = usecase_witness(year_start=self.year_start, year_end=self.year_end)
        witness_uc.study_name = self.study_name
        data_witness = witness_uc.setup_usecase()
        
        dspace = witness_uc.witness_uc.dspace 
        list_design_var_to_clean = ['red_meat_calories_per_day_ctrl', 'white_meat_calories_per_day_ctrl', 'vegetables_and_carbs_calories_per_day_ctrl', 'milk_and_eggs_calories_per_day_ctrl', 'forest_investment_array_mix', 'deforestation_investment_ctrl']

        # clean dspace
        dspace.drop(dspace.loc[dspace['variable'].isin(list_design_var_to_clean)].index, inplace=True)

        # clean dspace descriptor 
        dvar_descriptor = witness_uc.witness_uc.design_var_descriptor
        
        updated_dvar_descriptor = {k:v for k,v in dvar_descriptor.items() if k not in list_design_var_to_clean}
        updated_data = {f'{self.study_name}.{witness_uc.optim_name}.{witness_uc.coupling_name}.{witness_uc.extra_name}.assumptions_dict': {'compute_gdp': True,
                                                                'compute_climate_impact_on_gdp': True,
                                                                'activate_climate_effect_population': True,
                                                                'activate_pandemic_effects': True,
                                                                'invest_co2_tax_in_renewables': False
                                                               },
                        f'{self.study_name}.{witness_uc.optim_name}.design_space' : dspace,
                        f'{self.study_name}.{witness_uc.optim_name}.{witness_uc.coupling_name}.{witness_uc.witness_uc.designvariable_name}.design_var_descriptor': updated_dvar_descriptor}
        data_witness.append(updated_data)

        data_witness.append({
            f"{self.study_name}.{witness_uc.optim_name}.{witness_uc.coupling_name}.{witness_uc.extra_name}.ccs_price_percentage": 100.0,
            f"{self.study_name}.{witness_uc.optim_name}.{witness_uc.coupling_name}.{witness_uc.extra_name}.co2_damage_price_percentage": 100.0,
        })

        return data_witness


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
    design_space_out_var_name = uc_cls.ee.dm.get_all_namespaces_from_var_name("design_space_out")[0]
    dspace_end = uc_cls.ee.dm.get_value(design_space_out_var_name)
    design_space_last_ite_var_name = uc_cls.ee.dm.get_all_namespaces_from_var_name("design_space_last_ite")[0]
    dspace_last_ite = uc_cls.ee.dm.get_value(design_space_last_ite_var_name)
    max_ite_varname = uc_cls.ee.dm.get_all_namespaces_from_var_name("max_iter")[0]
    max_itee = uc_cls.ee.dm.get_value(max_ite_varname)
    import os

    path_dir = os.path.dirname(os.path.abspath(__file__))
    path_ds_end = os.path.join(path_dir, f"{uc_cls.study_name}_design_space_out_{max_itee}_iter.csv")
    path_ds_last_ite = os.path.join(path_dir, f"{uc_cls.study_name}_design_space_last_ite_{max_itee}_iter.csv")
    dspace_end.to_csv(path_ds_end)
    dspace_last_ite.to_csv(path_ds_last_ite)