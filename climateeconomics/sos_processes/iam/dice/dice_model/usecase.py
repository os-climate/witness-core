'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2023/11/03 Copyright 2023 Capgemini

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
from numpy import arange, zeros, append
from pandas import DataFrame

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):
    year_start = 2015
    year_end = GlossaryCore.YearEndDefault
    time_step = 5

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self, study_folder_path=None):
        setup_data_list = []
        # private values economics operator pyworld3
        dice_input = {}

        dice_input[f"{self.study_name}.{GlossaryCore.YearStart}"] = self.year_start
        dice_input[f"{self.study_name}.{GlossaryCore.YearEnd}"] = self.year_end
        dice_input[f"{self.study_name}.{GlossaryCore.TimeStep}"] = self.time_step

        dice_input[f"{self.study_name}.{'Carbon_emissions'}.{'init_land_emissions'}"] = 2.6
        dice_input[f"{self.study_name}.{'Carbon_emissions'}.{'decline_rate_land_emissions'}"] = .115
        dice_input[f"{self.study_name}.{'Carbon_emissions'}.{'init_cum_land_emisisons'}"] = 100.0
        dice_input[f"{self.study_name}.{'Carbon_emissions'}.{'init_gr_sigma'}"] = -0.0152
        dice_input[f"{self.study_name}.{'Carbon_emissions'}.{'decline_rate_decarbo'}"] = -0.001
        dice_input[f"{self.study_name}.{'Carbon_emissions'}.{'init_indus_emissions'}"] = 35.7
        dice_input[f"{self.study_name}.{'init_gross_output'}"] = 105.177
        dice_input[f"{self.study_name}.{'Carbon_emissions'}.{'init_cum_indus_emissions'}"] = 400.0

        dice_input[f"{self.study_name}.{'Damage'}.{'init_damag_int'}"] = 0.0
        dice_input[f"{self.study_name}.{'Damage'}.{'damag_int'}"] = 0.0
        dice_input[f"{self.study_name}.{'Damage'}.{'damag_quad'}"] = 0.0022
        dice_input[f"{self.study_name}.{'Damage'}.{'damag_expo'}"] = 2.0
        dice_input[f"{self.study_name}.{'Damage'}.{'exp_cont_f'}"] = 2.6
        dice_input[f"{self.study_name}.{'Damage'}.{'cost_backstop'}"] = 550.0
        dice_input[f"{self.study_name}.{'Damage'}.{'init_cost_backstop'}"] = .025
        dice_input[f"{self.study_name}.{'Damage'}.{'gr_base_carbonprice'}"] = .02
        dice_input[f"{self.study_name}.{'Damage'}.{'init_base_carbonprice'}"] = 2.0
        dice_input[f"{self.study_name}.{'Damage'}.{'tipping_point'}"] = False
        dice_input[f"{self.study_name}.{'Damage'}.{'tp_a1'}"] = 20.46
        dice_input[f"{self.study_name}.{'Damage'}.{'tp_a2'}"] = 2.0
        dice_input[f"{self.study_name}.{'Damage'}.{'tp_a3'}"] = 6.081
        dice_input[f"{self.study_name}.{'Damage'}.{'tp_a4'}"] = 6.754
        dice_input[f"{self.study_name}.{GlossaryCore.DamageToProductivity}"] = False
        dice_input[f"{self.study_name}.{GlossaryCore.FractionDamageToProductivityValue}"] = 0.30

        dice_input[f"{self.study_name}.{'Macroeconomics'}.{'productivity_start'}"] = 5.115
        dice_input[f"{self.study_name}.{'Macroeconomics'}.{'capital_start'}"] = 223.0
        dice_input[f"{self.study_name}.{'Macroeconomics'}.{'pop_start'}"] = 7403.0
        dice_input[f"{self.study_name}.{'Macroeconomics'}.{'output_elasticity'}"] = .300
        dice_input[f"{self.study_name}.{'Macroeconomics'}.{'popasym'}"] = 11500.0
        dice_input[f"{self.study_name}.{'Macroeconomics'}.{'population_growth'}"] = 0.134
        dice_input[f"{self.study_name}.{'Macroeconomics'}.{'productivity_gr_start'}"] = 0.076
        dice_input[f"{self.study_name}.{'Macroeconomics'}.{'decline_rate_tfp'}"] = 0.005
        dice_input[f"{self.study_name}.{'Macroeconomics'}.{'depreciation_capital'}"] = .100
        dice_input[f"{self.study_name}.{'init_rate_time_pref'}"] = .015
        dice_input[f"{self.study_name}.{'conso_elasticity'}"] = 1.45
        dice_input[f"{self.study_name}.{'Macroeconomics'}.{'lo_capital'}"] = 1.0
        dice_input[f"{self.study_name}.{'Macroeconomics'}.{'lo_conso'}"] = 2.0
        dice_input[f"{self.study_name}.{'Macroeconomics'}.{'lo_per_capita_conso'}"] = 0.01
        dice_input[f"{self.study_name}.{'Macroeconomics'}.{'saving_rate'}"] = 0.2

        dice_input[f"{self.study_name}.{'Temperature_change'}.{'init_temp_ocean'}"] = .00687
        dice_input[f"{self.study_name}.{'Temperature_change'}.{'init_temp_atmo'}"] = 0.85
        dice_input[f"{self.study_name}.{'Temperature_change'}.{'eq_temp_impact'}"] = 3.1
        dice_input[f"{self.study_name}.{'Temperature_change'}.{'init_forcing_nonco'}"] = 0.5
        dice_input[f"{self.study_name}.{'Temperature_change'}.{'hundred_forcing_nonco'}"] = 1.0
        dice_input[f"{self.study_name}.{'Temperature_change'}.{'climate_upper'}"] = 0.1005
        dice_input[f"{self.study_name}.{'Temperature_change'}.{'transfer_upper'}"] = 0.088
        dice_input[f"{self.study_name}.{'Temperature_change'}.{'transfer_lower'}"] = 0.025
        dice_input[f"{self.study_name}.{'Temperature_change'}.{'forcing_eq_co2'}"] = 3.6813
        dice_input[f"{self.study_name}.{'Temperature_change'}.{'lo_tocean'}"] = -1.0
        dice_input[f"{self.study_name}.{'Temperature_change'}.{'up_tatmo'}"] = 12.0
        dice_input[f"{self.study_name}.{'Temperature_change'}.{'up_tocean'}"] = 20.0

        dice_input[f"{self.study_name}.{'Utility.scaleone'}"] = 0.0302455265681763
        dice_input[f"{self.study_name}.{'Utility.scaletwo'}"] = -10993.704

        nb_per = round(
            (self.year_end - self.year_start) / self.time_step + 1)
        years = arange(self.year_start, self.year_end + 1, self.time_step)
        miu0 = 0.03
        dice_emissions = [0.0323, 0.0349, 0.0377, 0.0408, 0.0441, 0.0476, 0.0515,
                          0.0556, 0.0601, 0.0650, 0.0702, 0.0759, 0.0821, 0.0887, 0.0959, 0.1036, 0.1120]
        dice_emissions = append(miu0, dice_emissions)
        emissions_control_rate = append(
            dice_emissions, zeros(nb_per - len(dice_emissions)))
        emissions_control_rate = DataFrame(
            {'year': years, 'value': emissions_control_rate})
        dice_input[f"{self.study_name}.{'emissions_control_rate'}"] = emissions_control_rate

        # Only need to initialize one dataframe , apre run will compute the
        # other ones
        data = zeros(len(emissions_control_rate))
        df_eco = DataFrame({'year': years,
                            'saving_rate': data,
                            GlossaryCore.GrossOutput: data,
                            GlossaryCore.OutputNetOfDamage: data,
                            GlossaryCore.NetOutput: data,
                            GlossaryCore.PopulationValue: data,
                            GlossaryCore.Productivity: data,
                            GlossaryCore.ProductivityGrowthRate: data,
                            GlossaryCore.Consumption: data,
                            GlossaryCore.PerCapitaConsumption: data,
                            GlossaryCore.Capital: data,
                            GlossaryCore.InvestmentsValue: data,
                            'interest_rate': data},
                           index=arange(self.year_start, self.year_end + 1, self.time_step))

        dice_input[self.study_name + f'.{GlossaryCore.EconomicsDfValue}'] = df_eco

        setup_data_list.append(dice_input)

        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.test()
