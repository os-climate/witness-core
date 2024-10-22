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
from scipy.interpolate import interp1d
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore


class UtilityJacobianDiscTest(AbstractJacobianUnittest):

    def setUp(self):
        self.name = 'Test'
        self.model_name = 'utility'
        self.year_start =GlossaryCore.YearStartDefault
        self.year_end = GlossaryCore.YearEndDefaultTest
        self.years = np.arange(self.year_start, self.year_end + 1)
        self.year_range = self.year_end - self.year_start

        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}'}
        self.ee = ExecutionEngine(self.name)
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.utilitymodel.utilitymodel_discipline.UtilityModelDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        f = interp1d([self.year_start, self.year_start + 1, self.year_start +2, (self.year_start + self.year_end) / 2, self.year_end], [100, 100, 100, 200, 100])
        gdp_net = f(self.years)
        self.economics_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.GrossOutput: np.linspace(121, 91, len(self.years)),
            GlossaryCore.PerCapitaConsumption: np.linspace(12, 6, len(self.years)),
            GlossaryCore.OutputNetOfDamage: gdp_net,
        })

        self.population_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.PopulationValue: np.linspace(7886, 9550, len(self.years))
        })
        self.population_df.index = self.years

        energy_mean_price = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.EnergyPriceValue: np.linspace(200, 10, len(self.years))
        })

        self.values_dict = {f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                            f'{self.name}.{GlossaryCore.EconomicsDfValue}': self.economics_df,
                            f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                            f'{self.name}.{GlossaryCore.EnergyMeanPriceValue}': energy_mean_price}

        self.ee.load_study_from_input_dict(self.values_dict)


    def analytic_grad_entry(self):
        return [
            self.test_01_utility_analytic_grad_welfare,
        ]

    def test_01_utility_analytic_grad_welfare(self):
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__), filename='jacobian_utility_discipline_welfare.pkl', discipline=disc_techno, step=1e-15,local_data = disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyMeanPriceValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}'
                            ],
                            outputs=[f'{self.name}.{GlossaryCore.QuantityObjectiveValue}',
                                     f'{self.name}.{GlossaryCore.DecreasingGdpIncrementsObjectiveValue}',
                            ],
                            derr_approx='complex_step')

    def test_02_utility_analytic_grad_welfare_no_population_multiplication_in_obj(self):
        self.values_dict.update({
            f'{self.name}.{self.model_name}.multiply_obj_by_pop': False
        })
        self.ee.load_study_from_input_dict(self.values_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__), filename='jacobian_utility_discipline_welfare2.pkl', discipline=disc_techno, step=1e-15,local_data = disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyMeanPriceValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}'
                            ],
                            outputs=[f'{self.name}.{GlossaryCore.QuantityObjectiveValue}',
                                     f'{self.name}.{GlossaryCore.DecreasingGdpIncrementsObjectiveValue}',
                            ],
                            derr_approx='complex_step')
