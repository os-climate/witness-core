'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2025/01/10 Copyright 2025 Capgemini

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
from os.path import dirname

import numpy as np
import pandas as pd
from sostrades_optimization_plugins.models.test_class import GenericDisciplinesTestClass

from climateeconomics.glossarycore import GlossaryCore


class ForestryJacobianDiscTest(GenericDisciplinesTestClass):

    def setUp(self):
        self.name = 'Test'
        self.model_name = "Agriculture.Forestry"
        self.pickle_directory = dirname(__file__)
        self.ns_dict = {'ns_public': f'{self.name}',
                        GlossaryCore.NS_WITNESS: f'{self.name}',
                        GlossaryCore.NS_FUNCTIONS: f'{self.name}.{self.model_name}',
                        'ns_forestry': f'{self.name}.{self.model_name}',
                        'ns_sectors': f'{self.name}',
                        'ns_agriculture': f'{self.name}.{self.model_name}',
                        'ns_invest': f'{self.name}.{self.model_name}'}

    def get_inputs_dict(self) -> dict:
        year_start = GlossaryCore.YearStartDefault
        year_end = GlossaryCore.YearEndDefaultTest
        years = np.arange(year_start, year_end + 1, 1)
        crop_reduction_productivity_df = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.CropProductivityReductionName: np.linspace(0, 1.5, len(years))
        })

        years = np.arange(year_start, year_end + 1, 1)
        year_range = year_end - year_start + 1

        invest_df = pd.DataFrame({
            GlossaryCore.Years: years,
            'Reforestation': np.linspace(2, 10, year_range),
            'Deforestation': np.linspace(10, 1, year_range),
            'Managed wood': np.linspace(1, 10, year_range),
        })

        transport_df = pd.DataFrame({GlossaryCore.Years: years, "transport": 7.6})
        margin = pd.DataFrame({GlossaryCore.Years: years, 'margin': 110.})
        return {
            f'{self.name}.mdo_sectors_invest_level': 2,
            f'{self.name}.{GlossaryCore.YearStart}': year_start,
            f'{self.name}.{GlossaryCore.YearEnd}': year_end,
            f'{self.name}.{self.model_name}.{GlossaryCore.InvestmentDetailsDfValue}': invest_df,
            f'{self.name}.transport_cost': transport_df,
            f'{self.name}.margin': margin,
            f'{self.name}.{GlossaryCore.CropProductivityReductionName}': crop_reduction_productivity_df,
        }

    def test_forestry(self):
        self.jacobian_test = True
        self.override_dump_jacobian = True
        self.mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestryDiscipline'


if __name__ == "__main__":
    unittest.main()
