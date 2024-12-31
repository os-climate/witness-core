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
import unittest
from os.path import dirname

import numpy as np
import pandas as pd

from climateeconomics.core.tools.discipline_tester import discipline_test_function
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc import (
    ForestDiscipline,
)


class ForestJacobianDiscTest(unittest.TestCase):

    def setUp(self):
        self.name = 'Test'
        self.model_name = 'Forest'
        self.ns_dict = {'ns_public': f'{self.name}',
                   GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}.{self.model_name}',
                   'ns_forest': f'{self.name}.{self.model_name}',
                   'ns_agriculture': f'{self.name}.{self.model_name}',
                   'ns_invest': f'{self.name}.{self.model_name}'}

        year_start = GlossaryCore.YearStartDefault
        year_end = GlossaryCore.YearEndDefaultTest
        years = np.arange(year_start, year_end + 1, 1)
        crop_reduction_productivity_df = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.CropProductivityReductionName: np.linspace(0, 1.5, len(years))
        })

        years = np.arange(year_start, year_end + 1, 1)
        year_range = year_end - year_start + 1

        reforestation_invest_df = pd.DataFrame({
            GlossaryCore.Years: years,
            "reforestation_investment": np.linspace(2, 10, year_range)
        })
        deforest_invest_df = pd.DataFrame({
            GlossaryCore.Years: years,
            GlossaryCore.InvestmentsValue: np.linspace(10, 1, year_range)
        })

        mw_invest_df = pd.DataFrame({GlossaryCore.Years: years,
                                          GlossaryCore.InvestmentsValue: np.linspace(1, 10, year_range)})
        transport_df = pd.DataFrame({GlossaryCore.Years: years, "transport": 7.6})
        margin = pd.DataFrame({GlossaryCore.Years: years, 'margin': 110.})

        self.inputs_dict = {
            f'{self.name}.{GlossaryCore.YearStart}': year_start,
            f'{self.name}.{GlossaryCore.YearEnd}': year_end,
            f'{self.name}.{self.model_name}.deforestation_investment': deforest_invest_df,
            f'{self.name}.{self.model_name}.reforestation_investment': reforestation_invest_df,
            f'{self.name}.{self.model_name}.managed_wood_investment': mw_invest_df,
            f'{self.name}.transport_cost': transport_df,
            f'{self.name}.margin': margin,
            f'{self.name}.{GlossaryCore.CropProductivityReductionName}': crop_reduction_productivity_df,
        }


    def test_forest(self):
        discipline_test_function(
            module_path='climateeconomics.sos_wrapping.sos_wrapping_agriculture.forest.forest_disc.ForestDiscipline',
            model_name=self.model_name,
            name=self.name,
            jacobian_test=True,
            coupling_inputs=ForestDiscipline.coupling_inputs,
            coupling_outputs=ForestDiscipline.coupling_outputs,
            show_graphs=False,
            inputs_dict=self.inputs_dict,
            namespaces_dict=self.ns_dict,
            pickle_directory=dirname(__file__),
            pickle_name='jacobian_forest_autodiff.pkl',
            override_dump_jacobian=False
        )