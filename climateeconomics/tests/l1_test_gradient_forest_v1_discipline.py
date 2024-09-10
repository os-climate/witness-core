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
from os.path import dirname

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.core.core_forest.forest_v1 import Forest
from climateeconomics.glossarycore import GlossaryCore


class ForestJacobianDiscTest(AbstractJacobianUnittest):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_forest_analytic_grad
        ]

    def test_forest_analytic_grad(self):

        model_name = 'Test'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_forest': f'{self.name}.{model_name}',
                   'ns_invest': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_forest.forest_v1.forest_disc.ForestDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = 2050
        self.time_step = 1
        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1
        deforestation_surface = np.array(np.linspace(10, 100, year_range))
        self.deforestation_surface_df = pd.DataFrame(
            {GlossaryCore.Years: years, "deforested_surface": deforestation_surface})
        self.CO2_per_ha = 4000
        self.limit_deforestation_surface = 1000
        # GtCO2
        self.initial_emissions = 3.21
        forest_invest = np.linspace(2, 4, year_range)
        self.forest_invest_df = pd.DataFrame(
            {GlossaryCore.Years: years, "forest_investment": forest_invest})
        self.reforestation_cost_per_ha = 3800

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.TimeStep}': 1,
                       f'{self.name}.{model_name}.{Forest.LIMIT_DEFORESTATION_SURFACE}': self.limit_deforestation_surface,
                       f'{self.name}.{Forest.DEFORESTATION_SURFACE}': self.deforestation_surface_df,
                       f'{self.name}.{model_name}.{Forest.CO2_PER_HA}': self.CO2_per_ha,
                       f'{self.name}.{model_name}.{Forest.INITIAL_CO2_EMISSIONS}': self.initial_emissions,
                       f'{self.name}.{Forest.REFORESTATION_INVESTMENT}': self.forest_invest_df,
                       f'{self.name}.{model_name}.{Forest.REFORESTATION_COST_PER_HA}': self.reforestation_cost_per_ha,
                       }
        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_forest_v1_discipline.pkl',
                            discipline=disc_techno, local_data=disc_techno.local_data, step=1e-15, derr_approx='complex_step',
                            inputs=[
                                f'{self.name}.{Forest.DEFORESTATION_SURFACE}', f'{self.name}.{Forest.REFORESTATION_INVESTMENT}'],
                            outputs=[f'{self.name}.{Forest.FOREST_SURFACE_DF}',
                                     f'{self.name}.{Forest.CO2_EMITTED_FOREST_DF}'])
