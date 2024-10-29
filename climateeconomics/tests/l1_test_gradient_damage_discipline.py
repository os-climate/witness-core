'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/04-2023/11/03 Copyright 2023 Capgemini

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


class DamageJacobianDiscTest(AbstractJacobianUnittest):

    def setUp(self):
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        self.year_end = GlossaryCore.YearEndDefaultTest
        self.years = np.arange(GlossaryCore.YearStartDefault, self.year_end + 1, 1)

        self.temperature_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TempAtmo: np.linspace(-0.7, 6.6, len(self.years)),
        })

        self.damage_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.Damages: np.linspace(40, 60, len(self.temperature_df)),
            GlossaryCore.EstimatedDamages: np.linspace(40, 60, len(self.temperature_df)),
        })

        self.extra_co2_t_since_preindustrial = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.ExtraCO2EqSincePreIndustrialValue: np.linspace(100, 300, len(self.years))
        })
        self.model_name = 'Test'
        self.coupled_inputs = [f'{self.name}.{GlossaryCore.TemperatureDfValue}',
                               f'{self.name}.{GlossaryCore.ExtraCO2EqSincePreIndustrialValue}',
                               f'{self.name}.{GlossaryCore.DamageDfValue}'
                               ]

        self.coupled_outputs = [f'{self.name}.{GlossaryCore.DamageFractionDfValue}',
                                f'{self.name}.{GlossaryCore.CO2DamagePrice}',
                                f'{self.name}.{self.model_name}.{GlossaryCore.ExtraCO2tDamagePrice}'
                                ]

        self.ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_dashboard': f'{self.name}',
                   f'ns_{GlossaryCore.SectorIndustry.lower()}_emissions': self.name,
                   f'ns_{GlossaryCore.SectorServices.lower()}_emissions': self.name,
                   f'ns_{GlossaryCore.SectorAgriculture.lower()}_emissions': self.name,
                   f'ns_{GlossaryCore.SectorIndustry.lower()}_gdp': self.name,
                   f'ns_{GlossaryCore.SectorServices.lower()}_gdp': self.name,
                   f'ns_{GlossaryCore.SectorAgriculture.lower()}_gdp': self.name,
                   GlossaryCore.NS_SECTORS_POST_PROC_EMISSIONS: f'{self.name}',
                   GlossaryCore.NS_SECTORS_POST_PROC_GDP: f'{self.name}',
                   GlossaryCore.NS_REGIONALIZED_POST_PROC: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   GlossaryCore.NS_HOUSEHOLDS_EMISSIONS: f'{self.name}'}

    def analytic_grad_entry(self):
        return [
            self.test_damage_analytic_grad
        ]

    def test_damage_analytic_grad(self):
        self.ee.ns_manager.add_ns_def(self.ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.damagemodel.damagemodel_discipline.DamageDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.{self.model_name}.tipping_point': True,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.ExtraCO2EqSincePreIndustrialValue}': self.extra_co2_t_since_preindustrial,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': pd.DataFrame(
                           {GlossaryCore.Years: self.years, GlossaryCore.CO2Tax: np.linspace(50, 500, len(self.years))}),
                       f'{self.name}.{GlossaryCore.TemperatureDfValue}': self.temperature_df,
                       f'{self.name}.{self.model_name}.damage_constraint_factor': np.concatenate(
                           (np.linspace(0.5, 1, 15), np.asarray([1] * (len(self.years) - 15))))}

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()
        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_damage_discipline.pkl',
                            discipline=disc_techno, local_data=disc_techno.local_data,
                            step=1e-15,
                            inputs=self.coupled_inputs,
                            outputs=self.coupled_outputs,
                            derr_approx='complex_step')

    def test_damage_analytic_grad_wo_damage_on_climate(self):
        """
        Test analytic gradient with damage on climate deactivated
        """

        self.ee.ns_manager.add_ns_def(self.ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.damagemodel.damagemodel_discipline.DamageDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.{self.model_name}.tipping_point': True,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.ExtraCO2EqSincePreIndustrialValue}': self.extra_co2_t_since_preindustrial,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': pd.DataFrame(
                           {GlossaryCore.Years: self.years, GlossaryCore.CO2Tax: np.linspace(50, 500, len(self.years))}),
                       f'{self.name}.{GlossaryCore.TemperatureDfValue}': self.temperature_df,
                       f'{self.name}.{self.model_name}.damage_constraint_factor': np.concatenate(
                           (np.linspace(0.5, 1, 15), np.asarray([1] * (len(self.years) - 15)))),
                       f'{self.name}.assumptions_dict':
                           {'compute_gdp': True,
                            'compute_climate_impact_on_gdp': False,
                            'activate_climate_effect_population': False,
                            'activate_pandemic_effects': False,
                            }
                       }

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()
        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_damage_discipline_wo_damage_on_climate.pkl',
                            discipline=disc_techno, local_data=disc_techno.local_data,
                            step=1e-15,
                            inputs=self.coupled_inputs,
                            outputs=self.coupled_outputs,
                            derr_approx='complex_step')

    def test_damage_analytic_grad_dev_formula(self):
        self.ee.ns_manager.add_ns_def(self.ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.damagemodel.damagemodel_discipline.DamageDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.{self.model_name}.tipping_point': True,
                       f'{self.name}.co2_damage_price_dev_formula': True,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.ExtraCO2EqSincePreIndustrialValue}': self.extra_co2_t_since_preindustrial,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': pd.DataFrame(
                           {GlossaryCore.Years: self.years, GlossaryCore.CO2Tax: np.linspace(50, 500, len(self.years))}),
                       f'{self.name}.{GlossaryCore.TemperatureDfValue}': self.temperature_df,
                       f'{self.name}.{self.model_name}.damage_constraint_factor': np.concatenate(
                           (np.linspace(0.5, 1, 15), np.asarray([1] * (len(self.years) - 15))))}

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()
        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_damage_discipline_dev_formula.pkl',
                            discipline=disc_techno, local_data=disc_techno.local_data,
                            step=1e-15,
                            inputs=self.coupled_inputs,
                            outputs=self.coupled_outputs,
                            derr_approx='complex_step')

    def test_damage_analytic_grad_dice(self):
        self.ee.ns_manager.add_ns_def(self.ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.damagemodel.damagemodel_discipline.DamageDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.{self.model_name}.tipping_point': False,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{GlossaryCore.DamageDfValue}': self.damage_df,
                       f'{self.name}.{GlossaryCore.ExtraCO2EqSincePreIndustrialValue}': self.extra_co2_t_since_preindustrial,
                       f'{self.name}.{GlossaryCore.CO2TaxesValue}': pd.DataFrame(
                           {GlossaryCore.Years: self.years, GlossaryCore.CO2Tax: np.linspace(50, 500, len(self.years))}),
                       f'{self.name}.{GlossaryCore.TemperatureDfValue}': self.temperature_df,
                       f'{self.name}.{self.model_name}.damage_constraint_factor': np.concatenate(
                           (np.linspace(0.5, 1, 15), np.asarray([1] * (len(self.years) - 15))))}

        self.ee.load_study_from_input_dict(inputs_dict)
        self.ee.execute()
        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_damage_discipline_dice.pkl',
                            discipline=disc_techno, local_data=disc_techno.local_data,
                            step=1e-15,
                            inputs=self.coupled_inputs,
                            outputs=self.coupled_outputs,
                            derr_approx='complex_step')
