'''
Copyright 2022 Airbus SAS
Modifications on {} Copyright 2024 Capgemini
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
from energy_models.database_witness_energy import DatabaseWitnessEnergy
from energy_models.glossaryenergy import GlossaryEnergy
from sostrades_core.study_manager.study_manager import StudyManager

from climateeconomics.database.database_witness_core import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


class ClimateEconomicsStudyManager(StudyManager):
    '''
    Class that overloads study manager to define a specific check for climate economics usecases
    '''
    def should_be_lower(self, actual_value, ref_value, varname: str) -> str:
        msg = ''
        if actual_value > ref_value:
            msg = f"{varname>140} should be lower than {ref_value} but is not. Value = {actual_value}"
        return msg

    def should_be_greater(self, actual_value, ref_value, varname: str) -> str:
        msg = ''
        if actual_value < ref_value:
            msg = f"\n{varname:>140} should be greater than {ref_value} but is not. Value = {actual_value}"
        return msg

    @staticmethod
    def update_dataframes_with_year_star(values_dict: dict,  year_start: int) -> dict:
        """truncate all dataframe rows that are before year start"""
        year_start_varnames = list(filter(lambda x: f".{GlossaryCore.YearStart}" in x, values_dict.keys()))
        values_dict.update({varname: year_start for varname in year_start_varnames})

        values_dict_2023 = {}
        for key, value in values_dict.items():
            if isinstance(value, pd.DataFrame) and GlossaryCore.Years in value.columns:
                new_value = value.loc[value[GlossaryCore.Years] >= year_start]
                values_dict_2023[key] = new_value
        values_dict.update(values_dict_2023)
        return values_dict

    @staticmethod
    def get_share_invest_by_techno_of_total_energy_invest_for_coarse(selected_year: int):
        """
        returns the share invested in each witnes coarse techno relative to total invests in energy sector at the year selected
        """

        technos_coarse = []

        for stream, technos in GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT.items():
            technos_coarse.extend(technos['value'])
        total_energy_invest_selected_year = 0
        invests = []
        for techno in technos_coarse:
            a, b = DatabaseWitnessEnergy.get_techno_invest_before_year_start(techno_name=techno,
                                                                             year_start=selected_year + 1,
                                                                             construction_delay=1)
            invests.append(a['invest'].values[0])
            total_energy_invest_selected_year += a['invest'].values[0]


        share_invest_by_techno = np.round(np.array(invests) / total_energy_invest_selected_year * 100, 2)

        share_invest_by_techno = dict(zip(technos_coarse, share_invest_by_techno))
        return share_invest_by_techno

    @staticmethod
    def get_share_invest_in_eneryg_relative_to_gdp(selected_year: int):
        """returns the share invested in energy relative to total gdp"""

        technos_coarse = []

        for stream, technos in GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT.items():
            technos_coarse.extend(technos['value'])
        total_energy_invest_selected_year = 0
        invests = []
        for techno in technos_coarse:
            a, b = DatabaseWitnessEnergy.get_techno_invest_before_year_start(techno_name=techno,
                                                                             year_start=selected_year + 1,
                                                                             construction_delay=1)
            invests.append(a['invest'].values[0])
            total_energy_invest_selected_year += a['invest'].values[0]

        world_gdp = DatabaseWitnessCore.MacroInitGrossOutput.get_value_at_year(selected_year)

        percent_of_gdp = total_energy_invest_selected_year / 1000 / world_gdp * 100

        return np.round(percent_of_gdp, 2)
