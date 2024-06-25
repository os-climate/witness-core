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

from os.path import dirname, join

import pandas as pd
from energy_models.glossaryenergy import GlossaryEnergy
from energy_models.sos_processes.energy.MDA.energy_process_v0.usecase import (
    INVEST_DISC_NAME,
)

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev.usecase_witness_coarse_new import (
    Study as usecase_witness_mda,
)


class Study(ClimateEconomicsStudyManager):
    '''
    Usecase 1 mda run only, no optim, based on usecase_witness_coarse_new. Working assumptions:
    - Macro-model: N/A
    - Damage: N/A
    - Tax: N/A
    - invest: fossil
    '''

    def __init__(self, run_usecase=False, execution_engine=None, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, time_step=1):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step


    def setup_usecase(self, study_folder_path=None):
        witness_uc = usecase_witness_mda()
        witness_uc.study_name = self.study_name
        data_witness = witness_uc.setup_usecase()

        # update the assumption dict to setup the working assumptions
        updated_data = {f'{self.study_name}.assumptions_dict': {'compute_gdp': False,
                                                                'compute_climate_impact_on_gdp': False,
                                                                'activate_climate_effect_population': False,
                                                                'activate_pandemic_effects': False,
                                                                'invest_co2_tax_in_renewables': False
                                                               },
                        f"{self.study_name}.ccs_price_percentage": 0.0,
                        f"{self.study_name}.co2_damage_price_percentage": 0.0,
                        f"{self.study_name}.{'Macroeconomics'}.{'damage_to_productivity'}": False
                        }
        data_witness.append(updated_data)

        # Inputs were optimized manually through the sostrades GUI and saved in csv files  => recover inputs
        invest_percentage_gdp_df = pd.read_csv(join(dirname(__file__), 'uc1_percentage_of_gdp_energy_invest.csv'))
        invest_percentage_per_techno_df = pd.read_csv(join(dirname(__file__), 'uc1_techno_invest_percentage.csv'))

        # csv files are valid for years 2020 to 2100 => to be adapted if year range is smaller.
        #TODO: implement if year range is larger than 2020-2100
        invest_percentage_gdp_df = invest_percentage_gdp_df[
            (invest_percentage_gdp_df[GlossaryCore.Years] >= self.year_start) &
            (invest_percentage_gdp_df[GlossaryCore.Years] <= self.year_end)].reset_index(drop=True)
        invest_percentage_per_techno_df = invest_percentage_per_techno_df[
            (invest_percentage_per_techno_df[GlossaryCore.Years] >= self.year_start) &
            (invest_percentage_per_techno_df[GlossaryCore.Years] <= self.year_end)].reset_index(drop=True)

        data_witness.append(
            {
                f'{self.study_name}.{INVEST_DISC_NAME}.{GlossaryEnergy.EnergyInvestPercentageGDPName}': invest_percentage_gdp_df,
                f'{self.study_name}.{INVEST_DISC_NAME}.{GlossaryEnergy.TechnoInvestPercentageName}': invest_percentage_per_techno_df,
                }
        )

        return data_witness


if '__main__' == __name__:
    uc_cls = Study() #Study(run_usecase=True)
    #uc_cls.load_data()
    #uc_cls.run()
    uc_cls.test()

