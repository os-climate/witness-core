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

from os.path import dirname, join

import numpy as np
from pandas import read_csv
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore


class TemperatureJacobianDiscTest(AbstractJacobianUnittest):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_02_temperature_discipline_analytic_grad_DICE,
            self.test_03_temperature_discipline_analytic_grad_FUND_myhre,
            self.test_03_1_temperature_discipline_analytic_grad_FUND_Meinshausen
        ]

    def test_02_temperature_discipline_analytic_grad_DICE(self):

        self.model_name = 'temperature'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange_v2.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_ally = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))
        # Take only from year start value
        ghg_cycle_df = carboncycle_df_ally[carboncycle_df_ally[GlossaryCore.Years] >= GlossaryCore.YearStartDefault]

        ghg_cycle_df[GlossaryCore.CO2Concentration] = ghg_cycle_df['ppm']
        ghg_cycle_df[GlossaryCore.CH4Concentration] = ghg_cycle_df['ppm'] * 1222/296
        ghg_cycle_df[GlossaryCore.N2OConcentration] = ghg_cycle_df['ppm'] * 296/296
        ghg_cycle_df = ghg_cycle_df[[GlossaryCore.Years, GlossaryCore.CO2Concentration, GlossaryCore.CH4Concentration, GlossaryCore.N2OConcentration]]

        # put manually the index
        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        ghg_cycle_df.index = years

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': GlossaryCore.YearStartDefault,
                       f'{self.name}.{GlossaryCore.YearEnd}': GlossaryCore.YearEndDefault,
                       f'{self.name}.{GlossaryCore.GHGCycleDfValue}': ghg_cycle_df,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.{self.model_name}.temperature_model': 'DICE',
                       }

        self.ee.load_study_from_input_dict(values_dict)

        # self.ee.execute()

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_temperature_v2_discipline_DICE.pkl',
                            discipline=disc_techno,local_data = disc_techno.local_data,
                            step=1e-15,
                            inputs=[f'{self.name}.{GlossaryCore.GHGCycleDfValue}'],
                            outputs=[f'{self.name}.{GlossaryCore.TemperatureDfValue}',
                                     f'{self.name}.temperature_constraint',
                                     f'{self.name}.{self.model_name}.forcing_detail_df',
                                     ],
                            derr_approx='complex_step')

    def test_03_temperature_discipline_analytic_grad_FUND_myhre(self):

        self.model_name = 'temperature'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange_v2.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_ally = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))
        # Take only from year start value
        ghg_cycle_df = carboncycle_df_ally[carboncycle_df_ally[GlossaryCore.Years] >= GlossaryCore.YearStartDefault]

        ghg_cycle_df[GlossaryCore.CO2Concentration] = ghg_cycle_df['ppm']
        ghg_cycle_df[GlossaryCore.CH4Concentration] = ghg_cycle_df['ppm'] * 1222/296
        ghg_cycle_df[GlossaryCore.N2OConcentration] = ghg_cycle_df['ppm'] * 296/296
        ghg_cycle_df = ghg_cycle_df[[GlossaryCore.Years, GlossaryCore.CO2Concentration, GlossaryCore.CH4Concentration, GlossaryCore.N2OConcentration]]

        # put manually the index
        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        ghg_cycle_df.index = years

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': GlossaryCore.YearStartDefault,
                       f'{self.name}.{GlossaryCore.YearEnd}': GlossaryCore.YearEndDefault,
                       f'{self.name}.{GlossaryCore.GHGCycleDfValue}': ghg_cycle_df,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.{self.model_name}.temperature_model': 'FUND',
                       f'{self.name}.{self.model_name}.forcing_model': 'Myhre',
                       }

        self.ee.load_study_from_input_dict(values_dict)

        # self.ee.execute()

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_temperature_v2_discipline_Myhre.pkl',
                            discipline=disc_techno,local_data = disc_techno.local_data,
                            step=1e-15,
                            inputs=[f'{self.name}.{GlossaryCore.GHGCycleDfValue}'],
                            outputs=[f'{self.name}.{GlossaryCore.TemperatureDfValue}',
                                     f'{self.name}.temperature_constraint',
                                     f'{self.name}.{self.model_name}.forcing_detail_df',
                                     ],
                            derr_approx='complex_step')

    def test_03_1_temperature_discipline_analytic_grad_FUND_Meinshausen(self):

        self.model_name = 'temperature'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange_v2.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_ally = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))
        # Take only from year start value
        ghg_cycle_df = carboncycle_df_ally[carboncycle_df_ally[GlossaryCore.Years] >= GlossaryCore.YearStartDefault]

        ghg_cycle_df[GlossaryCore.CO2Concentration] = ghg_cycle_df['ppm']
        ghg_cycle_df[GlossaryCore.CH4Concentration] = ghg_cycle_df['ppm'] * 1222/296
        ghg_cycle_df[GlossaryCore.N2OConcentration] = ghg_cycle_df['ppm'] * 296/296
        ghg_cycle_df = ghg_cycle_df[[GlossaryCore.Years, GlossaryCore.CO2Concentration, GlossaryCore.CH4Concentration, GlossaryCore.N2OConcentration]]

        # put manually the index
        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        ghg_cycle_df.index = years

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': GlossaryCore.YearStartDefault,
                       f'{self.name}.{GlossaryCore.YearEnd}': GlossaryCore.YearEndDefault,
                       f'{self.name}.{GlossaryCore.GHGCycleDfValue}': ghg_cycle_df,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.{self.model_name}.temperature_model': 'FUND',
                       f'{self.name}.{self.model_name}.forcing_model': 'Meinshausen',
                       }

        self.ee.load_study_from_input_dict(values_dict)

        # self.ee.execute()

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_temperature_v2_discipline_Meinshausen.pkl',
                            discipline=disc_techno,local_data = disc_techno.local_data,
                            step=1e-15,
                            inputs=[f'{self.name}.{GlossaryCore.GHGCycleDfValue}'],
                            outputs=[f'{self.name}.{GlossaryCore.TemperatureDfValue}',
                                     f'{self.name}.temperature_constraint',
                                     f'{self.name}.{self.model_name}.forcing_detail_df',
                                     ],
                            derr_approx='complex_step')

    def _test_04_temperature_discipline_analytic_grad_etminan(self):

        self.model_name = 'temperature'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange_v2.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_ally = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))
        # Take only from year start value
        ghg_cycle_df = carboncycle_df_ally[carboncycle_df_ally[GlossaryCore.Years] >= GlossaryCore.YearStartDefault]

        ghg_cycle_df[GlossaryCore.CO2Concentration] = ghg_cycle_df['ppm']
        ghg_cycle_df[GlossaryCore.CH4Concentration] = ghg_cycle_df['ppm'] * 1222/296
        ghg_cycle_df[GlossaryCore.N2OConcentration] = ghg_cycle_df['ppm'] * 296/296
        ghg_cycle_df = ghg_cycle_df[[GlossaryCore.Years, GlossaryCore.CO2Concentration, GlossaryCore.CH4Concentration, GlossaryCore.N2OConcentration]]

        # put manually the index
        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        ghg_cycle_df.index = years

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': GlossaryCore.YearStartDefault,
                       f'{self.name}.{GlossaryCore.YearEnd}': GlossaryCore.YearEndDefault,
                       f'{self.name}.{GlossaryCore.GHGCycleDfValue}': ghg_cycle_df,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.{self.model_name}.temperature_model': 'DICE',
                       f'{self.name}.{self.model_name}.forcing_model': 'Etminan',
                       }

        self.ee.load_study_from_input_dict(values_dict)

        # self.ee.execute()

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_temperature_v2_discipline_etminan.pkl',
                            discipline=disc_techno,local_data = disc_techno.local_data,
                            step=1e-15,
                            inputs=[f'{self.name}.{GlossaryCore.GHGCycleDfValue}'],
                            outputs=[f'{self.name}.{self.model_name}.forcing_detail_df',
                                     f'{self.name}.{GlossaryCore.TemperatureDfValue}',
                                     f'{self.name}.temperature_constraint'],
                            derr_approx='complex_step')

    def _test_05_temperature_discipline_analytic_grad_meinshausen(self):

        self.model_name = 'temperature'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange_v2.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_ally = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))
        # Take only from year start value
        ghg_cycle_df = carboncycle_df_ally[carboncycle_df_ally[GlossaryCore.Years] >= GlossaryCore.YearStartDefault]

        ghg_cycle_df[GlossaryCore.CO2Concentration] = ghg_cycle_df['ppm']
        ghg_cycle_df[GlossaryCore.CH4Concentration] = ghg_cycle_df['ppm'] * 1222/296
        ghg_cycle_df[GlossaryCore.N2OConcentration] = ghg_cycle_df['ppm'] * 296/296
        ghg_cycle_df = ghg_cycle_df[[GlossaryCore.Years, GlossaryCore.CO2Concentration, GlossaryCore.CH4Concentration, GlossaryCore.N2OConcentration]]

        # put manually the index
        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        ghg_cycle_df.index = years

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': GlossaryCore.YearStartDefault,
                       f'{self.name}.{GlossaryCore.YearEnd}': GlossaryCore.YearEndDefault,
                       f'{self.name}.{GlossaryCore.GHGCycleDfValue}': ghg_cycle_df,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.{self.model_name}.temperature_model': 'DICE',
                       f'{self.name}.{self.model_name}.forcing_model': 'Meinshausen', }

        self.ee.load_study_from_input_dict(values_dict)

        # self.ee.execute()

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        self.check_jacobian(location=dirname(__file__),
                            filename='jacobian_temperature_v2_discipline_Meinshausen.pkl',
                            discipline=disc_techno,local_data = disc_techno.local_data,
                            step=1e-15,
                            inputs=[f'{self.name}.{GlossaryCore.GHGCycleDfValue}'],
                            outputs=[f'{self.name}.{self.model_name}.forcing_detail_df',
                                     f'{self.name}.{GlossaryCore.TemperatureDfValue}',
                                     f'{self.name}.temperature_constraint'],
                            derr_approx='complex_step')

    def _test_06_temperature_discipline_analytic_grad_etminan_lower_atmo_conc(self):

        self.model_name = 'temperature'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange_v2.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_all = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))

        carboncycle_df_y = carboncycle_df_all[carboncycle_df_all[GlossaryCore.Years] >= GlossaryCore.YearStartDefault]
        carboncycle_df = carboncycle_df_y[[GlossaryCore.Years, 'atmo_conc']]
        # put manually the index
        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        carboncycle_df.index = years

        values_dict = {f'{self.name}.{GlossaryCore.YearStart}': GlossaryCore.YearStartDefault,
                       f'{self.name}.{GlossaryCore.YearEnd}': GlossaryCore.YearEndDefault,
                       f'{self.name}.{GlossaryCore.CarbonCycleDfValue}': carboncycle_df,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.{self.model_name}.forcing_model': 'Etminan',
                       f'{self.name}.{self.model_name}.pre_indus_co2_concentration_ppm': 41000.}

        self.ee.load_study_from_input_dict(values_dict)

        # self.ee.execute()

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_temperature_v2_discipline_etminan_lower.pkl',
                            local_data = disc_techno.local_data,
                            discipline=disc_techno, step=1e-10, inputs=[f'{self.name}.{GlossaryCore.CarbonCycleDfValue}'],
                            outputs=[f'{self.name}.{self.model_name}.forcing_detail_df', f'{self.name}.{GlossaryCore.TemperatureDfValue}', f'{self.name}.temperature_objective', f'{self.name}.temperature_constraint'], output_column='CO2 forcing', derr_approx='finite_differences')


if '__main__' == __name__:
    cls = TemperatureJacobianDiscTest()
    cls.setUp()
    cls.test_03_temperature_discipline_analytic_grad_FUND_myhre()
