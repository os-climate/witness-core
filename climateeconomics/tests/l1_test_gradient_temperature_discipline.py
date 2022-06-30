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

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class TemperatureJacobianDiscTest(AbstractJacobianUnittest):
    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_temperature_discipline_analytic_grad_integral_option,
            self.test_temperature_discipline_analytic_grad_last_temperature_option
        ]

    def test_temperature_discipline_analytic_grad_integral_option(self):

        self.__temperature_discipline_analytic_grad('integral')

    def test_temperature_discipline_analytic_grad_last_temperature_option(self):

        self.__temperature_discipline_analytic_grad('last_temperature')

    def __temperature_discipline_analytic_grad(self, temperature_obj_option):

        self.model_name = 'temperature'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_ref': f'{self.name}',
                   'ns_public': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_all = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))

        carboncycle_df_y = carboncycle_df_all[carboncycle_df_all['years'] >= 2020]
        carboncycle_df = carboncycle_df_y[['years', 'atmo_conc']]
        # put manually the index
        years = np.arange(2020, 2101, 1)
        carboncycle_df.index = years

        values_dict = {f'{self.name}.year_start': 2020,
                       f'{self.name}.year_end': 2100,
                       f'{self.name}.time_step': 1,
                       f'{self.name}.carboncycle_df': carboncycle_df,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.temperature_obj_option': temperature_obj_option,
                       f'{self.name}.{self.model_name}.forcing_model': 'DICE'}

        self.ee.load_study_from_input_dict(values_dict)

        # self.ee.execute()

        disc_techno = self.ee.root_process.sos_disciplines[0]

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_temperature_discipline_{temperature_obj_option}.pkl', discipline=disc_techno, step=1e-15, inputs=[f'{self.name}.carboncycle_df'],
                            outputs=[f'{self.name}.temperature_df', f'{self.name}.temperature_objective', f'{self.name}.temperature_constraint'], derr_approx='complex_step')

    def test_03_temperature_discipline_analytic_grad_myhre(self):

        self.model_name = 'temperature'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_all = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))

        carboncycle_df_y = carboncycle_df_all[carboncycle_df_all['years'] >= 2020]
        carboncycle_df = carboncycle_df_y[['years', 'atmo_conc']]
        # put manually the index
        years = np.arange(2020, 2101, 1)
        carboncycle_df.index = years

        values_dict = {f'{self.name}.year_start': 2020,
                       f'{self.name}.year_end': 2100,
                       f'{self.name}.time_step': 1,
                       f'{self.name}.carboncycle_df': carboncycle_df,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.{self.model_name}.forcing_model': 'Myhre', }

        self.ee.load_study_from_input_dict(values_dict)

        # self.ee.execute()

        disc_techno = self.ee.root_process.sos_disciplines[0]

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_temperature_discipline_Myhre.pkl', discipline=disc_techno, step=1e-15, inputs=[f'{self.name}.carboncycle_df'],
                            outputs=[f'{self.name}.temperature_df', f'{self.name}.temperature_objective',  f'{self.name}.temperature_constraint'], derr_approx='complex_step')

    def _test_04_temperature_discipline_analytic_grad_etminan(self):

        self.model_name = 'temperature'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_all = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))

        carboncycle_df_y = carboncycle_df_all[carboncycle_df_all['years'] >= 2020]
        carboncycle_df = carboncycle_df_y[['years', 'atmo_conc']]
        # put manually the index
        years = np.arange(2020, 2101, 1)
        carboncycle_df.index = years

        values_dict = {f'{self.name}.year_start': 2020,
                       f'{self.name}.year_end': 2100,
                       f'{self.name}.time_step': 1,
                       f'{self.name}.carboncycle_df': carboncycle_df,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.{self.model_name}.forcing_model': 'Etminan', }

        self.ee.load_study_from_input_dict(values_dict)

        # self.ee.execute()

        disc_techno = self.ee.root_process.sos_disciplines[0]

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_temperature_discipline_etminan.pkl', discipline=disc_techno, step=1e-15, inputs=[f'{self.name}.carboncycle_df'],
                            outputs=[f'{self.name}.{self.model_name}.forcing_detail_df', f'{self.name}.temperature_df', f'{self.name}.temperature_objective', f'{self.name}.temperature_constraint'], derr_approx='complex_step')

    def test_05_temperature_discipline_analytic_grad_meinshausen(self):

        self.model_name = 'temperature'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_all = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))

        carboncycle_df_y = carboncycle_df_all[carboncycle_df_all['years'] >= 2020]
        carboncycle_df = carboncycle_df_y[['years', 'atmo_conc']]
        # put manually the index
        years = np.arange(2020, 2101, 1)
        carboncycle_df.index = years

        values_dict = {f'{self.name}.year_start': 2020,
                       f'{self.name}.year_end': 2100,
                       f'{self.name}.time_step': 1,
                       f'{self.name}.carboncycle_df': carboncycle_df,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.{self.model_name}.forcing_model': 'Meinshausen', }

        self.ee.load_study_from_input_dict(values_dict)

        # self.ee.execute()

        disc_techno = self.ee.root_process.sos_disciplines[0]

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_temperature_discipline_Meinshausen.pkl', discipline=disc_techno, step=1e-15, inputs=[f'{self.name}.carboncycle_df'],
                            outputs=[f'{self.name}.temperature_df', f'{self.name}.temperature_objective', f'{self.name}.temperature_constraint'], derr_approx='complex_step')

    def _test_06_temperature_discipline_analytic_grad_etminan_lower_atmo_conc(self):

        self.model_name = 'temperature'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.tempchange.tempchange_discipline.TempChangeDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        data_dir = join(dirname(__file__), 'data')
        carboncycle_df_all = read_csv(
            join(data_dir, 'carbon_cycle_data_onestep.csv'))

        carboncycle_df_y = carboncycle_df_all[carboncycle_df_all['years'] >= 2020]
        carboncycle_df = carboncycle_df_y[['years', 'atmo_conc']]
        # put manually the index
        years = np.arange(2020, 2101, 1)
        carboncycle_df.index = years

        values_dict = {f'{self.name}.year_start': 2020,
                       f'{self.name}.year_end': 2100,
                       f'{self.name}.time_step': 1,
                       f'{self.name}.carboncycle_df': carboncycle_df,
                       f'{self.name}.alpha': 0.5,
                       f'{self.name}.{self.model_name}.forcing_model': 'Etminan',
                       f'{self.name}.{self.model_name}.pre_indus_co2_concentration_ppm': 41000.}

        self.ee.load_study_from_input_dict(values_dict)

        # self.ee.execute()

        disc_techno = self.ee.root_process.sos_disciplines[0]

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_temperature_discipline_etminan_lower.pkl', discipline=disc_techno, step=1e-10, inputs=[f'{self.name}.carboncycle_df'],
                            outputs=[f'{self.name}.{self.model_name}.forcing_detail_df', f'{self.name}.temperature_df', f'{self.name}.temperature_objective', f'{self.name}.temperature_constraint'], output_column='CO2 forcing', derr_approx='finite_differences')


if '__main__' == __name__:
    cls = TemperatureJacobianDiscTest()
    #AbstractJacobianUnittest.DUMP_JACOBIAN = True
    cls.setUp()
    cls.test_04_temperature_discipline_analytic_grad_etminan()
