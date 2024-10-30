'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/07-2023/11/03 Copyright 2023 Capgemini

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

import logging
from os.path import dirname

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore


class GHGCycleJacobianDiscTest(AbstractJacobianUnittest):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        self.years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1)

        self.ghg_emissions_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalCO2Emissions: np.linspace(35, 0, len(self.years)),
            GlossaryCore.TotalCH4Emissions: np.linspace(35, 0, len(self.years)) * 0.3 / 40,
            GlossaryCore.TotalN2OEmissions: np.linspace(35, 0, len(self.years)) * 0.008 / 40,
        })

    def analytic_grad_entry(self):
        return [
            self.test_execute,
        ]

    def test_execute(self):

        self.model_name = 'GHGCycle'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)
        self.ee.logger.setLevel(logging.DEBUG)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.ghgcycle.ghgcycle_discipline.GHGCycleDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        values_dict = {f'{self.name}.{GlossaryCore.GHGEmissionsDfValue}': self.ghg_emissions_df}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_ghg_cycle_discipline1.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data = disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.GHGEmissionsDfValue}'],
                            outputs=[f'{self.name}.{GlossaryCore.GHGCycleDfValue}',
                                     f'{self.name}.gwp20_objective',
                                     f'{self.name}.gwp100_objective',
                                     f'{self.name}.rockstrom_limit_constraint',
                                     f'{self.name}.minimum_ppm_constraint',
                                     f'{self.name}.{GlossaryCore.ExtraCO2EqSincePreIndustrialValue}',])
