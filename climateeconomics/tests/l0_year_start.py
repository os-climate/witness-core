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
import unittest

import numpy as np
from energy_models.database_witness_energy import DatabaseWitnessEnergy
from energy_models.glossaryenergy import GlossaryEnergy

from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.database.collected_data import ColectedData, HeavyCollectedData
from climateeconomics.glossarycore import GlossaryCore


class YearStartTest(unittest.TestCase):
    """
    This test is created to assert that first point of coarse optimization investment variables is not activated and
    is set to the correct value (invest at 2020
    """

    def test_year_start_witness_coarse(self):
        """print output message if you want to see which years are ok, and what inputs are missing at which year"""
        tested_years = np.arange(2018, 2024)
        missing_var_dict = {year: self._test_year(year, techno_dict=GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT) for year in tested_years}
        valid_years = []
        output_msg = ""
        for year in tested_years:
            if len(missing_var_dict[year]) > 0:
                msg = f"Not OK, {len(missing_var_dict[year])} missing inputs : {missing_var_dict[year]}"
            else:
                msg = "OK"
                valid_years.append(year)
            output_msg += f"{year}: {msg}\n"
        print(output_msg)
        self.assertIn(GlossaryCore.YearStartDefault, valid_years,
                      msg=f"There are some missing inputs for the selected year start ({GlossaryCore.YearStartDefault})")

    def test_year_start_witness_full(self):
        """print output message if you want to see which years are ok, and what inputs are missing at which year"""
        tested_years = np.arange(2018, 2024)
        missing_var_dict = {year: self._test_year(year, techno_dict=GlossaryEnergy.DEFAULT_TECHNO_DICT_DEV) for year in tested_years}
        valid_years = []
        output_msg = ""
        for year in tested_years:
            if len(missing_var_dict[year]) > 0:
                msg = f"Not OK, {len(missing_var_dict[year])} missing inputs : {missing_var_dict[year]}"
            else:
                msg = "OK"
                valid_years.append(year)
            output_msg += f"{year}: {msg}\n"

        #print(output_msg)
        self.assertIn(GlossaryCore.YearStartDefault, valid_years,
                      msg=f"There are some missing inputs for the selected year start ({GlossaryCore.YearStartDefault})")

    def _get_techno_list_from_techno_dict(self, techno_dict: dict):
        """gathers all technos in a list from a techno dict"""
        techno_list = []
        for stream, stream_info in techno_dict.items():
            techno_list.extend(stream_info['value'])

        return techno_list

    def _test_year_witness_core(self, year: int) -> list[str]:
        """Test all values that are critical in witness core"""
        missing_variables = []
        database_attributes = dir(DatabaseWitnessCore)
        for attribute_name in database_attributes:
            attribute_value = getattr(DatabaseWitnessCore, attribute_name)
            if isinstance(attribute_value, ColectedData) and attribute_value.critical_at_year_start:
                if isinstance(attribute_value, HeavyCollectedData):
                    if not attribute_value.is_available_at_year(year=year):
                        missing_variables.append(attribute_name)
                else:
                    if attribute_value.year_value != year:
                        missing_variables.append(attribute_name)

        # forest invest before year start :
        is_available = DatabaseWitnessCore.get_forest_invest_before_year_start(year_start=year, construction_delay=3,
                                                                               is_available_at_year=True)
        if not is_available:
            missing_variables.append("Forest.managed_wood_invest_before_year_start")

        return missing_variables

    def _test_year_witness_energy(self, year: int, techno_dict: dict) -> list[str]:
        missing_variables = []
        techno_list = self._get_techno_list_from_techno_dict(techno_dict=techno_dict)

        # initial production
        for techno_name in techno_list:
            is_available = DatabaseWitnessEnergy.get_techno_prod(techno_name=techno_name, year=year,
                                                                 is_available_at_year=True)
            if not is_available:
                missing_variables.append(f"{techno_name}.initial_production")

        # invest_before_year_start
        for techno_name in techno_list:
            is_available = DatabaseWitnessEnergy.get_techno_invest_before_year_start(techno_name=techno_name,
                                                                                     construction_delay=
                                                                                     GlossaryEnergy.TechnoConstructionDelayDict[
                                                                                         techno_name], year_start=year,
                                                                                     is_available_at_year=True)
            if not is_available:
                missing_variables.append(f"{techno_name}.invest_before_year_start")

        # Initial plants age distribution
        for techno_name in techno_list:
            is_available = DatabaseWitnessEnergy.get_techno_age_distrib_factor(techno_name=techno_name, year=year,
                                                                               is_available_at_year=True)
            if not is_available:
                missing_variables.append(f"{techno_name}.{GlossaryCore.InitialPlantsAgeDistribFactor}")

        return missing_variables

    def _test_year(self, year: int, techno_dict: dict) -> list[str]:
        """test if a given year is ok"""
        missing_variables = self._test_year_witness_core(year=year)
        missing_variables.extend(self._test_year_witness_energy(year=year, techno_dict=techno_dict))
        return missing_variables

