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


class InvestJacobianDiscTest(AbstractJacobianUnittest):
    #AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [
            self.test_invest_objective_grad,
        ]

    def test_invest_objective(self):

        self.model_name = 'invest_connexion'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.invest_discipline.InvestDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        invest_ref = 1055.0    # G$
        self.year_start = 2020
        self.year_end = 2050
        years = np.arange(self.year_start, self.year_end + 1)
        invest = np.ones(len(years)) * invest_ref
        invest[0] = invest_ref
        for i in range(1, len(years)):
            invest[i] = (1.0 - 0.0253) * invest[i - 1]

        energy_investment_macro = pd.DataFrame(
            {'years': years, 'energy_investment': invest})
        energy_investment = pd.DataFrame(
            {'years': years, 'energy_investment': invest * 2})
        values_dict = {f'{self.name}.energy_investment_macro': energy_investment_macro,
                       f'{self.name}.energy_investment': energy_investment,
                       f'{self.name}.{self.model_name}.invest_norm': 18.4}

        self.ee.dm.set_values_from_dict(values_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.sos_disciplines[0]
        filter = disc_techno.get_chart_filter_list()
        graph_list = disc_techno.get_post_processing_list(filter)

#         for graph in graph_list:
#             graph.to_plotly().show()

    def test_invest_objective_grad(self):

        self.model_name = 'invest_connexion'
        ns_dict = {'ns_witness': f'{self.name}',
                   'ns_energy_mix': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_witness.invest_discipline.InvestDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        invest_ref = 1055.0    # G$
        self.year_start = 2020
        self.year_end = 2050
        years = np.arange(self.year_start, self.year_end + 1)
        invest = np.ones(len(years)) * invest_ref
        invest[0] = invest_ref
        for i in range(1, len(years)):
            invest[i] = (1.0 - 0.0253) * invest[i - 1]

        energy_investment_macro = pd.DataFrame(
            {'years': years, 'energy_investment': invest})
        energy_investment = pd.DataFrame(
            {'years': years, 'energy_investment': invest * 2})
        values_dict = {f'{self.name}.energy_investment_macro': energy_investment_macro,
                       f'{self.name}.energy_investment': energy_investment,
                       f'{self.name}.{self.model_name}.invest_norm': 18.4}

        self.ee.dm.set_values_from_dict(values_dict)

        disc_techno = self.ee.root_process.sos_disciplines[0]
        #AbstractJacobianUnittest.DUMP_JACOBIAN = True
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_invest_discipline.pkl', discipline=disc_techno, step=1e-4, threshold=1.0e-5,
                            inputs=[f'{self.name}.energy_investment',
                                    f'{self.name}.energy_investment_macro'],
                            outputs=[f'{self.name}.invest_objective'], derr_approx='finite_differences')
