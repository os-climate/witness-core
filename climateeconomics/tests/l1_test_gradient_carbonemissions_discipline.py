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
from sostrades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class CarbonEmissionsJacobianDiscTest(AbstractJacobianUnittest):
    # AbstractJacobianUnittest.DUMP_JACOBIAN = True
    # np.set_printoptions(threshold=np.inf)

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        self.years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1)
        self.economics_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.GrossOutput: np.linspace(121, 91, len(self.years)),
        })

        self.energy_supply_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalCO2Emissions: np.linspace(35, 0, len(self.years))
        })

        self.co2_emissions_ccus_Gt = pd.DataFrame({
            GlossaryCore.Years: self.years,
            'carbon_storage Limited by capture (Gt)': 0.02
        })

        self.CO2_emissions_by_use_sources = pd.DataFrame({
            GlossaryCore.Years: self.years,
            'CO2 from energy mix (Gt)': 0.0,
            'carbon_capture from energy mix (Gt)': 0.0,
            'Total CO2 by use (Gt)': 20.0,
            'Total CO2 from Flue Gas (Gt)': 3.2
        })

        self.CO2_emissions_by_use_sinks = pd.DataFrame({
            GlossaryCore.Years: self.years,
            'CO2 removed by energy mix (Gt)': 0.0
        })


        self.co2_emissions_needed_by_energy_mix = pd.DataFrame({
            GlossaryCore.Years: self.years,
            'carbon_capture needed by energy mix (Gt)': 0.0
        })

        self.CO2_emitted_forest = pd.DataFrame({
            GlossaryCore.Years: self.years,
            'emitted_CO2_evol': 0.04,
            'emitted_CO2_evol_cumulative': np.cumsum(np.linspace(0.01, 0.10, len(self.years))) + 3.21
        })


    def analytic_grad_entry(self):
        return [
            self.test_carbon_emissions_analytic_grad,
            self.test_co2_objective_limit_grad
        ]

    def test_carbon_emissions_analytic_grad(self):

        self.model_name = 'carbonemission'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   GlossaryCore.NS_CCS: f'{self.name}',
                   'ns_energy': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.carbonemissions.carbonemissions_discipline.CarbonemissionsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()


        values_dict = {f'{self.name}.{GlossaryCore.EconomicsDfValue}': self.economics_df,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': self.energy_supply_df,
                       f'{self.name}.CO2_land_emissions': self.CO2_emitted_forest,
                       f'{self.name}.co2_emissions_ccus_Gt': self.co2_emissions_ccus_Gt,
                       f'{self.name}.CO2_emissions_by_use_sources': self.CO2_emissions_by_use_sources,
                       f'{self.name}.CO2_emissions_by_use_sinks': self.CO2_emissions_by_use_sinks,
                       f'{self.name}.co2_emissions_needed_by_energy_mix': self.co2_emissions_needed_by_energy_mix}

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_carbon_emission_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data = disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                    f'{self.name}.CO2_emissions_by_use_sources',
                                    f'{self.name}.CO2_land_emissions',
                                    f'{self.name}.CO2_emissions_by_use_sinks', f'{self.name}.co2_emissions_needed_by_energy_mix', f'{self.name}.co2_emissions_ccus_Gt'],
                            outputs=[f'{self.name}.{GlossaryCore.CO2EmissionsDfValue}',
                                     f'{self.name}.CO2_objective', f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}'])

    def test_co2_objective_limit_grad(self):

        self.model_name = 'carbonemission'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   GlossaryCore.NS_CCS: f'{self.name}',
                   'ns_energy': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.carbonemissions.carbonemissions_discipline.CarbonemissionsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        energy_supply_df = pd.DataFrame({
            GlossaryCore.Years: self.years,
            GlossaryCore.TotalCO2Emissions: np.linspace(0, -3000, len(self.years)),
        })

        values_dict = {f'{self.name}.{GlossaryCore.EconomicsDfValue}': self.economics_df,
                       f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': energy_supply_df,
                       f'{self.name}.CO2_land_emissions': self.CO2_emitted_forest,
                       f'{self.name}.co2_emissions_ccus_Gt': self.co2_emissions_ccus_Gt,
                       f'{self.name}.CO2_emissions_by_use_sources': self.CO2_emissions_by_use_sources,
                       f'{self.name}.CO2_emissions_by_use_sinks': self.CO2_emissions_by_use_sinks,
                       f'{self.name}.co2_emissions_needed_by_energy_mix': self.co2_emissions_needed_by_energy_mix}

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_co2_objective_limit.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data = disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.EconomicsDfValue}',
                                    f'{self.name}.CO2_emissions_by_use_sources',
                                    f'{self.name}.CO2_land_emissions',
                                    f'{self.name}.CO2_emissions_by_use_sinks', f'{self.name}.co2_emissions_needed_by_energy_mix', f'{self.name}.co2_emissions_ccus_Gt'],
                            outputs=[f'{self.name}.{GlossaryCore.CO2EmissionsDfValue}',
                                     f'{self.name}.CO2_objective', f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}'])
