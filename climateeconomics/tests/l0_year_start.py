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

from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.database.collected_data import ColectedData, HeavyCollectedData
from climateeconomics.glossarycore import GlossaryCore


class YearStartTest(unittest.TestCase):
    """
    This test is created to assert that first point of coarse optimization investment variables is not activated and
    is set to the correct value (invest at 2020
    """

    def setUp(self):
        pass

    def _test_year(self, year: int):
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

        return missing_variables

    def test_year_start(self):
        tested_years = np.arange(2018, 2024)
        missing_var_dict = {year: self._test_year(year) for year in tested_years}
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

