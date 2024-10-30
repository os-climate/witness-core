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

from os.path import dirname

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore


class SectorizedUtilityJacobianDiscTest(AbstractJacobianUnittest):

    def setUp(self):
        self.name = 'Test'
        self.model_name = GlossaryCore.Consumption
        self.year_start = GlossaryCore.YearStartDefault
        self.year_end = GlossaryCore.YearEndDefault
        self.years = np.arange(self.year_start, self.year_end + 1)
        self.year_range = self.year_end - self.year_start

        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_SECTORS: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}'}

        self.ee = ExecutionEngine(self.name)
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.utility.sectorized_utility_discipline.SectorizedUtilityDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1)

        self.population_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.PopulationValue: np.linspace(7886, 9550, len(self.years))
        })

        self.energy_mean_price = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.EnergyPriceValue: np.linspace(200, 10, len(self.years))
        })

        # Sectorized Consumption
        self.sector_list = GlossaryCore.SectorsPossibleValues
        self.sectorized_consumption_df = pd.DataFrame({GlossaryCore.Years: self.years})
        for sector in self.sector_list:
            self.sectorized_consumption_df[sector] = 1.0
        np.random.seed(42)
        economics_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.GrossOutput: 0,
            GlossaryCore.OutputNetOfDamage: np.random.uniform(0, 150, len(self.years)),
            GlossaryCore.Capital: 0,
        })
        self.values_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                            f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                            f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                            f'{self.name}.{GlossaryCore.EconomicsDfValue}': economics_df,
                            f'{self.name}.{GlossaryCore.EnergyMeanPriceValue}': self.energy_mean_price,
                            f'{self.name}.{GlossaryCore.SectorizedConsumptionDfValue}': self.sectorized_consumption_df}

        self.ee.load_study_from_input_dict(self.values_dict)

    def analytic_grad_entry(self):
        return []

    def test_01_sectorization_gradients(self):
        """
        Test the gradients of the sectorized outputs
        """
        self.ee.execute()

        # self.override_dump_jacobian = True

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_sectorized_utility_discipline_sectors_obj.pkl',
                            discipline=disc_techno, step=1e-15, local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EnergyMeanPriceValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                    f'{self.name}.{GlossaryCore.SectorizedConsumptionDfValue}'],
                            outputs=[f'{self.name}.{sector}.{GlossaryCore.UtilityObjectiveName}' for sector in
                                     self.sector_list] + [
                                        f'{self.name}.{GlossaryCore.DecreasingGdpIncrementsObjectiveValue}', ],
                            derr_approx='complex_step')
