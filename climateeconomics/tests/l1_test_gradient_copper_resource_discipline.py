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

from os.path import join, dirname
import unittest
import numpy as np
import pandas as pd
from pandas import read_csv
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class CopperResourceJacobianDiscTest(AbstractJacobianUnittest):
    """
    Copper resource jacobian test class
    """
    # np.set_printoptions(threshold=np.inf)

    # AbstractJacobianUnittest.DUMP_JACOBIAN = True

    def analytic_grad_entry(self):
        return [
            self.test_copper_resource_analytic_grad,
            self.test_copper_resource_damand_variable_analytic_grad
        ]

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.year_start = 2020
        self.year_end = 2100
    

        self.lifespan = 5

        data_dir = join(dirname(__file__), 'data')

        self.energy_copper_demand_df = read_csv(
            join(data_dir, 'all_demand_from_energy_mix.csv'), usecols=['years', 'copper_resource'])
        self.energy_copper_variable_demand_df = read_csv(
            join(data_dir, 'all_demand_variable.csv'), usecols=['years', 'copper_resource'])
        self.consumed_copper_df = read_csv(
            join(data_dir, 'copper_resource_consumed_data.csv'))

        # part to adapt lenght to the year range

        self.energy_copper_demand_df = self.energy_copper_demand_df.loc[self.energy_copper_demand_df['years']
                                                                        >= self.year_start]
        self.energy_copper_demand_df = self.energy_copper_demand_df.loc[self.energy_copper_demand_df['years']
                                                                        <= self.year_end]
        self.energy_copper_variable_demand_df = self.energy_copper_variable_demand_df.loc[self.energy_copper_variable_demand_df['years']
                                                                                          >= self.year_start]
        self.energy_copper_variable_demand_df = self.energy_copper_variable_demand_df.loc[self.energy_copper_variable_demand_df['years']
                                                                                          <= self.year_end]

    def test_copper_resource_analytic_grad(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        self.model_name = 'copper_resource'

        ns_dict = {'ns_public': f'{self.name}',
                   'ns_witness': f'{self.name}.{self.model_name}',
                   'ns_functions': f'{self.name}.{self.model_name}',
                   'ns_copper_resource': f'{self.name}.{self.model_name}',
                   'ns_resource': f'{self.name}.{self.model_name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.core.core_resources.models.copper_resource.copper_resource_disc.CopperResourceDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.{self.model_name}.resources_demand': self.energy_copper_demand_df,
                    #    f'{self.name}.{self.model_name}.lifespan': self.lifespan,
                    #    f'{self.name}.{self.model_name}.resource_consumed_data'
                    #    : self.consumed_copper_df,
                       }

        self.ee.load_study_from_input_dict(inputs_dict)

        disc_techno = self.ee.root_process.sos_disciplines[0]
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_copper_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            inputs=[
                                f'{self.name}.{self.model_name}.resources_demand'],
                            outputs=[f'{self.name}.{self.model_name}.resource_stock',
                                     f'{self.name}.{self.model_name}.resource_price',
                                     f'{self.name}.{self.model_name}.use_stock',
                                     f'{self.name}.{self.model_name}.predictable_production',
                                     f'{self.name}.{self.model_name}.recycled_production'
                                     ])
        
    def test_copper_resource_damand_variable_analytic_grad(self):
        
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        self.model_name = 'copper_resource'

        ns_dict = {'ns_public': f'{self.name}',
                   'ns_witness': f'{self.name}.{self.model_name}',
                   'ns_functions': f'{self.name}.{self.model_name}',
                   'ns_copper_resource': f'{self.name}.{self.model_name}',
                   'ns_resource': f'{self.name}.{self.model_name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.core.core_resources.models.copper_resource.copper_resource_disc.CopperResourceDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.year_start': self.year_start,
                       f'{self.name}.year_end': self.year_end,
                       f'{self.name}.{self.model_name}.resources_demand': self.energy_copper_variable_demand_df,
                    #    f'{self.name}.{self.model_name}.lifespan': self.lifespan,
                    #    f'{self.name}.{self.model_name}.resource_consumed_data' : self.consumed_copper_df,
                       }
        self.ee.load_study_from_input_dict(inputs_dict)

        disc_techno = self.ee.root_process.sos_disciplines[0]
        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_copper_demand_variable_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',
                            inputs=[
                                f'{self.name}.{self.model_name}.resources_demand'],
                            outputs=[f'{self.name}.{self.model_name}.resource_stock',
                                     f'{self.name}.{self.model_name}.resource_price',
                                     f'{self.name}.{self.model_name}.use_stock',
                                     f'{self.name}.{self.model_name}.predictable_production',
                                     f'{self.name}.{self.model_name}.recycled_production'
                                     ])


if __name__ == "__main__":
    unittest.main()
