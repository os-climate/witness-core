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

from climateeconomics.glossarycore import GlossaryCore


class GHGEmissionsJacobianDiscTest(AbstractJacobianUnittest):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_carbon_emissions_analytic_grad
        ]

    def test_carbon_emissions_analytic_grad(self):

        self.model_name = 'agriculture_emissions'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_agriculture': f'{self.name}',
                   }
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_emissions.agriculture_emissions.agriculture_emissions_discipline.AgricultureEmissionsDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()
        year_start = GlossaryCore.YearStartDefault
        year_end = GlossaryCore.YearEndDefaultTest
        years = np.arange(year_start, year_end + 1)

        CO2_land_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                           'emitted_CO2_evol_cumulative': np.linspace(0., 0.7, len(years))})
        N2O_land_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                           'emitted_N2O_evol_cumulative': np.linspace(0., 0.4, len(years)),
                                           })
        CH4_land_emissions = pd.DataFrame({GlossaryCore.Years: years,
                                           'emitted_CH4_evol_cumulative': np.linspace(0., 0.5, len(years)),
                                           })

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': year_end,
                       f'{self.name}.{GlossaryCore.techno_list}': ['Crop', GlossaryCore.Forestry],
                       f'{self.name}.Crop.CO2_land_emission_df': CO2_land_emissions,
                       f'{self.name}.Forestry.CO2_land_emission_df': CO2_land_emissions,
                       f'{self.name}.Crop.CH4_land_emission_df': CH4_land_emissions,
                       f'{self.name}.Crop.N2O_land_emission_df': N2O_land_emissions,
                       }

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_agriculture_ghg_emission_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data = disc_techno.local_data,
                            inputs=[f'{self.name}.Crop.CO2_land_emission_df',
                                    f'{self.name}.Forestry.CO2_land_emission_df',
                                    f'{self.name}.Crop.CH4_land_emission_df',
                                    f'{self.name}.Crop.N2O_land_emission_df'],
                            outputs=[f'{self.name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2)}',
                                     f'{self.name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CH4)}',
                                     f'{self.name}.{GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.N2O)}'
                                     ])
