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

from os.path import join, dirname

import numpy as np
from pandas import read_csv

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class CarboncycleJacobianDiscTest(AbstractJacobianUnittest):
    # AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_execute,
            self.test_execute_2
        ]

    def test_execute(self):

        self.model_name = 'carboncycle'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_ref': f'{self.name}',
                   'ns_public': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.carboncycle.carboncycle_discipline.CarbonCycleDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')

        emission_df_all = read_csv(
            join(data_dir, 'co2_emissions_onestep.csv'))

        emission_df_y = emission_df_all[emission_df_all[GlossaryCore.Years] >= 2020][[GlossaryCore.Years,
                                                                           'total_emissions', 'cum_total_emissions']]

        # put manually the index
        years = np.arange(2020, 2101)
        emission_df_y.index = years

        values_dict = {f'{self.name}.{GlossaryCore.CO2EmissionsDfValue}': emission_df_y}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_carbon_cycle_discipline1.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data = disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.CO2EmissionsDfValue}'],
                            outputs=[f'{self.name}.{GlossaryCore.CarbonCycleDfValue}',
                                     f'{self.name}.ppm_objective',
                                     f'{self.name}.rockstrom_limit_constraint',
                                     f'{self.name}.minimum_ppm_constraint'])

    def test_execute_2(self):
        # test limit for max for lower_ocean_conc / upper_ocean_conc /
        # atmo_conc
        self.model_name = 'carboncycle'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.carboncycle.carboncycle_discipline.CarbonCycleDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')

        emission_df_all = read_csv(
            join(data_dir, 'co2_emissions_onestep.csv'))

        emission_df_y = emission_df_all[emission_df_all[GlossaryCore.Years] >= 2020][[GlossaryCore.Years,
                                                                           'total_emissions', 'cum_total_emissions']]

        # put manually the index
        years = np.arange(2020, 2101)
        emission_df_y.index = years

        values_dict = {f'{self.name}.{GlossaryCore.CO2EmissionsDfValue}': emission_df_y}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_carbon_cycle_discipline2.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step', local_data = disc_techno.local_data,
                            inputs=[f'{self.name}.{GlossaryCore.CO2EmissionsDfValue}'],
                            outputs=[f'{self.name}.{GlossaryCore.CarbonCycleDfValue}',
                                     f'{self.name}.ppm_objective',
                                     f'{self.name}.rockstrom_limit_constraint',
                                     f'{self.name}.minimum_ppm_constraint'])
