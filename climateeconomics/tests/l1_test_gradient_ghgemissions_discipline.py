'''
Copyright 2022 Airbus SAS

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
import pandas as pd
from os.path import join, dirname
from pandas import DataFrame, read_csv

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class GHGEmissionsJacobianDiscTest(AbstractJacobianUnittest):
    # AbstractJacobianUnittest.DUMP_JACOBIAN = True
    # np.set_printoptions(threshold=np.inf)

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_carbon_emissions_analytic_grad
        ]

    def test_carbon_emissions_analytic_grad(self):

        self.model_name = 'ghgemission'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_energy_mix': f'{self.name}',
                   'ns_agriculture': f'{self.name}',
                   'ns_ref': f'{self.name}',
                   'ns_ccs': f'{self.name}',
                   'ns_energy': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_emissions.ghgemissions.ghgemissions_discipline.GHGemissionsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()
        year_start = 2020
        year_end = 2100
        years = np.arange(year_start, year_end + 1)
        GHG_total_energy_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                                   GlossaryCore.TotalCO2Emissions: np.linspace(37., 10., len(years)),
                                                   'Total N2O emissions': np.linspace(1.7e-3, 5.e-4, len(years)),
                                                   'Total CH4 emissions': np.linspace(0.17, 0.01, len(years))})
        CO2_land_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                           'Crop': np.linspace(0., 0., len(years)),
                                           'Forest': np.linspace(3., 4., len(years))})
        CH4_land_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                           'Crop': np.linspace(0., 0., len(years)),
                                           'Forest': np.linspace(3., 4., len(years))})
        N2O_land_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                           'Crop': np.linspace(0., 0., len(years)),
                                           'Forest': np.linspace(3., 4., len(years))})

        CO2_indus_emissions_df = pd.DataFrame({GlossaryCore.Years: years,
                                               'indus_emissions': np.linspace(1., 2., len(years))})
        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': year_end,
                       f'{self.name}.CO2_land_emissions': CO2_land_emissions,
                       f'{self.name}.CH4_land_emissions': CH4_land_emissions,
                       f'{self.name}.N2O_land_emissions': N2O_land_emissions,
                       f'{self.name}.CO2_indus_emissions_df': CO2_indus_emissions_df,
                       f'{self.name}.GHG_total_energy_emissions': GHG_total_energy_emissions, }

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_ghg_emission_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data = disc_techno.local_data,
                            inputs=[f'{self.name}.CO2_land_emissions',
                                    f'{self.name}.CH4_land_emissions',
                                    f'{self.name}.N2O_land_emissions',
                                    f'{self.name}.CO2_indus_emissions_df',
                                    f'{self.name}.GHG_total_energy_emissions'],
                            outputs=[f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}',
                                     f'{self.name}.GHG_emissions_df'])
