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

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)


class ConsumptionJacobianDiscTest(AbstractJacobianUnittest):

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
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   GlossaryCore.NS_SECTORS: f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_GHGEMISSIONS: f'{self.name}'}

        self.ee = ExecutionEngine(self.name)
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.sectorized_utility.sectorized_utility_discipline.SectorizedUtilityDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1)
        self.economics_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.GrossOutput: np.linspace(121, 91, len(self.years)),
            GlossaryCore.OutputNetOfDamage: np.linspace(121, 91, len(self.years)),
            GlossaryCore.PerCapitaConsumption: 0.,
        })
        self.population_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.PopulationValue: np.linspace(7886, 9550, len(self.years))
        })

        self.energy_mean_price = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.EnergyPriceValue: np.linspace(200, 10, len(self.years))
        })

        self.residential_energy_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalProductionValue: np.linspace(200, 10, len(self.years))
        })

        self.investment_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.InvestmentsValue: np.full(len(self.years), 10.0)
        })

        # Sectorized Consumption
        self.sector_list = GlossaryCore.SectorsPossibleValues
        self.sectorized_consumption_df = pd.DataFrame({GlossaryCore.Years: self.years})
        for sector in self.sector_list:
            self.sectorized_consumption_df[sector] = 1.0

        self.values_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                            f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                            f'{self.name}.{GlossaryCore.EconomicsDfValue}': self.economics_df,
                            f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                            f'{self.name}.{GlossaryCore.EnergyMeanPriceValue}': self.energy_mean_price,
                            f'{self.name}.{GlossaryCore.ResidentialEnergyConsumptionDfValue}': self.residential_energy_df,
                            f'{self.name}.{GlossaryCore.InvestmentDfValue}': self.investment_df,
                            f'{self.name}.{GlossaryCore.AllSectorsDemandDfValue}': self.sectorized_consumption_df}

        self.ee.load_study_from_input_dict(self.values_dict)

    def analytic_grad_entry(self):
        return [
            self.test_01_consumption_analytic_grad_welfare,
            self.test_02_consumption_analytic_grad_last_utility,
            self.test_03_consumption_with_low_economy
        ]

    def test_01_consumption_analytic_grad_welfare(self):
        np.set_printoptions(threshold=np.inf)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__), filename='jacobian_consumption_discipline_welfare.pkl',
                            discipline=disc_techno, step=1e-15, local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyMeanPriceValue}',
                                    f'{self.name}.{GlossaryCore.ResidentialEnergyConsumptionDfValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.InvestmentDfValue}'],
                            outputs=[f'{self.name}.{GlossaryCore.UtilityDfValue}',
                                     f'{self.name}.{GlossaryCore.WelfareObjective}',
                                     f'{self.name}.min_utility_objective',
                                     f'{self.name}.{GlossaryCore.NegativeWelfareObjective}'],
                            derr_approx='complex_step')

    def test_02_consumption_analytic_grad_last_utility(self):
        """
        Test the second option of the objective function
        """

        self.values_dict[f'{self.name}.welfare_obj_option'] = 'last_utility'

        self.ee.load_study_from_input_dict(self.values_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__), filename='jacobian_consumption_discipline_last_utility.pkl',
                            discipline=disc_techno, step=1e-15, local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyMeanPriceValue}',
                                    f'{self.name}.{GlossaryCore.ResidentialEnergyConsumptionDfValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.InvestmentDfValue}'],
                            outputs=[f'{self.name}.{GlossaryCore.UtilityDfValue}',
                                     f'{self.name}.{GlossaryCore.WelfareObjective}',
                                     f'{self.name}.min_utility_objective',
                                     f'{self.name}.{GlossaryCore.NegativeWelfareObjective}'],
                            derr_approx='complex_step')

    def test_03_consumption_with_low_economy(self):
        economics_df = self.economics_df
        economics_df[GlossaryCore.OutputNetOfDamage] = self.economics_df[GlossaryCore.OutputNetOfDamage] / 2
        np.set_printoptions(threshold=np.inf)
        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.EconomicsDfValue}': economics_df,
                       f'{self.name}.{GlossaryCore.PopulationDfValue}': self.population_df,
                       f'{self.name}.{GlossaryCore.EnergyPriceValue}': self.energy_mean_price,
                       f'{self.name}.{GlossaryCore.ResidentialEnergyConsumptionDfValue}': self.residential_energy_df,
                       f'{self.name}.{GlossaryCore.InvestmentShareGDPValue}': self.investment_df}

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__), filename='jacobian_consumption_low_economy.pkl',
                            discipline=disc_techno,
                            step=1e-15, local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyMeanPriceValue}',
                                    f'{self.name}.{GlossaryCore.ResidentialEnergyConsumptionDfValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.InvestmentDfValue}'],
                            outputs=[f'{self.name}.{GlossaryCore.UtilityDfValue}',
                                     f'{self.name}.{GlossaryCore.WelfareObjective}',
                                     f'{self.name}.min_utility_objective',
                                     f'{self.name}.{GlossaryCore.NegativeWelfareObjective}'],
                            derr_approx='complex_step')

    def test_04_sectorization_gradients(self):
        """
        Test the gradients of the sectorized outputs
        """
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__), filename='jacobian_sectorized_utility_discipline_sectors_obj.pkl',
                            discipline=disc_techno, step=1e-15, local_data=disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                    f'{self.name}.{GlossaryCore.EnergyMeanPriceValue}',
                                    f'{self.name}.{GlossaryCore.ResidentialEnergyConsumptionDfValue}',
                                    f'{self.name}.{GlossaryCore.PopulationDfValue}',
                                    f'{self.name}.{GlossaryCore.InvestmentDfValue}'],
                            outputs=[f'{self.name}.{sector}.{GlossaryCore.UtilityObjective}' for sector in self.sector_list],
                            derr_approx='complex_step')