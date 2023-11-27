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

from climateeconomics.core.tools.ClimateEconomicsStudyManager import ClimateEconomicsStudyManager
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev.usecase_witness_coarse_new import \
    Study as usecase_witness_mda
from energy_models.database_witness_energy import DatabaseWitnessEnergy
from energy_models.glossaryenergy import GlossaryEnergy
from energy_models.sos_processes.energy.MDA.energy_process_v0.usecase import INVEST_DISC_NAME


class Study(ClimateEconomicsStudyManager):
    '''
    Usecase 5, mda run only, no optim, based on usecase_witness_coarse_new. Working assumptions:
    - Macro-model: compute GDP
    - Damage: activated on population, GDP
    - Tax: N/A
    - invest: mix fossil renewable, with CCS = CCS_2020
    '''

    def __init__(self, run_usecase=False, execution_engine=None, year_start=2020, year_end=2100, time_step=1):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step

    def setup_usecase(self):
        witness_uc = usecase_witness_mda()
        witness_uc.study_name = self.study_name
        data_witness = witness_uc.setup_usecase()

        # update the assumption dict to setup the working assumptions
        updated_data = {f'{self.study_name}.assumptions_dict': {'compute_gdp': True,
                                                                'compute_climate_impact_on_gdp': True,
                                                                'activate_climate_effect_population': True,
                                                                'invest_co2_tax_in_renewables': False
                                                               },
                        f"{self.study_name}.ccs_price_percentage": 0.0,
                        f"{self.study_name}.co2_damage_price_percentage": 0.0,
                        }
        data_witness.append(updated_data)

        # update of the energy investment values from the template usecase and csv file of investment
        study_v0 = witness_uc.dc_energy.study_v0
        invest_percentage_gdp = pd.DataFrame(data={GlossaryCore.Years: study_v0.years,
                                                        GlossaryEnergy.EnergyInvestPercentageGDPName: np.linspace(
                                                            2., 2., len(study_v0.years))})

        # The energy_investment are split in FossilSimpleTechno, 'RenewableSimpleTechno' and 'CarbonCaptureAndStorageTechno'
        # but CarbonCaptureAndStorageTechno must be split into
        # 1) carbon capture 'direct_air_capture.DirectAirCaptureTechno', 'flue_gas_capture.FlueGasTechno' and
        # 2) carbone storage 'CarbonStorageTechno'
        percentage_CarbonStorageTechno = 50.
        percentage_DirectAirCaptureTechno = 25.

        dbwitness = DatabaseWitnessEnergy()
        invest_percentage_per_techno_df = dbwitness.data_invest_steps_scenario
        # data_invest is defined between 2019 and 2100
        invest_percentage_per_techno_df = invest_percentage_per_techno_df.loc[
                (invest_percentage_per_techno_df[GlossaryCore.Years] >= self.year_start) &
                (invest_percentage_per_techno_df[GlossaryCore.Years] <= self.year_end)].reset_index(drop=True)

        invest_percentage_per_techno_df[GlossaryEnergy.CarbonStorageTechno] = \
            invest_percentage_per_techno_df[GlossaryEnergy.CarbonCaptureAndStorageTechno] * \
            percentage_CarbonStorageTechno / 100.

        invest_percentage_per_techno_df[GlossaryEnergy.DirectAirCapture] = \
            invest_percentage_per_techno_df[GlossaryEnergy.CarbonCaptureAndStorageTechno] * \
            percentage_DirectAirCaptureTechno / 100.

        invest_percentage_per_techno_df[GlossaryEnergy.FlueGasCapture] = \
            invest_percentage_per_techno_df[GlossaryEnergy.CarbonCaptureAndStorageTechno] - \
            invest_percentage_per_techno_df[GlossaryEnergy.CarbonStorageTechno] - \
            invest_percentage_per_techno_df[GlossaryEnergy.DirectAirCapture]

        invest_percentage_per_techno = invest_percentage_per_techno_df.loc[:,
                                                invest_percentage_per_techno_df.columns !=
                                                GlossaryEnergy.CarbonCaptureAndStorageTechno]

        data_witness.append(
            {
                f'{self.study_name}.{INVEST_DISC_NAME}.{GlossaryEnergy.EnergyInvestPercentageGDPName}': invest_percentage_gdp,
                f'{self.study_name}.{INVEST_DISC_NAME}.{GlossaryEnergy.TechnoInvestPercentageName}': invest_percentage_per_techno,
                }
        )

        return data_witness


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
    #uc_cls.test()

