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


class PolicyDiscTest(AbstractJacobianUnittest):
    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_policy_analytic_grad
        ]

    def test_policy_analytic_grad(self):

        self.model_name = 'policy'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.policymodel.policy_discipline.PolicyDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        years = np.arange(2020, 2101)
        CCS_price = pd.DataFrame(
            {'years': years, 'ccs_price_per_tCO2': np.linspace(100, 900, len(years))})
        CO2_damage = pd.DataFrame(
            {'years': years, 'CO2_damage_price': np.linspace(300, 700, len(years))})

        values_dict = {f'{self.name}.CCS_price': CCS_price,
                       f'{self.name}.CO2_damage_price': CO2_damage,
                       f'{self.name}.ccs_price_percentage': 50.,
                       f'{self.name}.co2_damage_price_percentage': 70.
                       }

        self.ee.dm.set_values_from_dict(values_dict)
        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_policy_discipline.pkl', discipline=disc, inputs=[f'{self.name}.CCS_price', f'{self.name}.CO2_damage_price'],
                            outputs=[f'{self.name}.CO2_taxes'], step=1e-15, derr_approx='complex_step')

    def test_policy_analytic_grad_2(self):

        self.model_name = 'policy'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.policymodel.policy_discipline.PolicyDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        years = np.arange(2020, 2101)
        CCS_price = pd.DataFrame(
            {'years': years, 'ccs_price_per_tCO2': np.linspace(900, 900, len(years))})
        CO2_damage = pd.DataFrame(
            {'years': years, 'CO2_damage_price': np.linspace(300, 700, len(years))})

        values_dict = {f'{self.name}.CCS_price': CCS_price,
                       f'{self.name}.CO2_damage_price': CO2_damage,
                       f'{self.name}.ccs_price_percentage': 50.,
                       f'{self.name}.co2_damage_price_percentage': 70.
                       }

        self.ee.dm.set_values_from_dict(values_dict)
        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_policy_discipline2.pkl', discipline=disc, inputs=[f'{self.name}.CCS_price', f'{self.name}.CO2_damage_price'],
                            outputs=[f'{self.name}.CO2_taxes'], step=1e-15, derr_approx='complex_step')

    def test_policy_analytic_grad_3(self):

        self.model_name = 'policy'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_public': f'{self.name}',
                   'ns_ref': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.policymodel.policy_discipline.PolicyDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        years = np.arange(2020, 2101)
        CCS_price = pd.DataFrame(
            {'years': years, 'ccs_price_per_tCO2': np.linspace(-100, -900, len(years))})
        CO2_damage = pd.DataFrame(
            {'years': years, 'CO2_damage_price': np.linspace(-300, -700, len(years))})

        values_dict = {f'{self.name}.CCS_price': CCS_price,
                       f'{self.name}.CO2_damage_price': CO2_damage,
                       f'{self.name}.ccs_price_percentage': 50.,
                       f'{self.name}.co2_damage_price_percentage': 60.
                       }

        self.ee.dm.set_values_from_dict(values_dict)
        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_policy_discipline3.pkl', discipline=disc, inputs=[f'{self.name}.CCS_price', f'{self.name}.CO2_damage_price'],
                            outputs=[f'{self.name}.CO2_taxes'], step=1e-15, derr_approx='complex_step')
