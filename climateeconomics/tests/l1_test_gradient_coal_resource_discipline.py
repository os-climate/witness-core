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

from pandas import read_csv
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore


class CoalResourceJacobianDiscTest(AbstractJacobianUnittest):
    """
    Coal resource jacobian test class
    """

    def analytic_grad_entry(self):
        return [
            self.test_coal_resource_analytic_grad,
            self.test_coal_resource_damand_variable_analytic_grad
        ]

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.year_start =GlossaryCore.YearStartDefault
        self.year_end = GlossaryCore.YearEndDefault

        data_dir = join(dirname(__file__), 'data')

        self.energy_coal_demand_df = read_csv(
            join(data_dir, 'all_demand_from_energy_mix.csv'), usecols=[GlossaryCore.Years,'coal_resource'])
        self.energy_coal_variable_demand_df = read_csv(
            join(data_dir, 'all_demand_variable.csv'), usecols=[GlossaryCore.Years,'coal_resource'])
        # part to adapt lenght to the year range

        self.energy_coal_demand_df = self.energy_coal_demand_df.loc[self.energy_coal_demand_df[GlossaryCore.Years]
                                                                    >= self.year_start]
        self.energy_coal_demand_df= self.energy_coal_demand_df.loc[self.energy_coal_demand_df[GlossaryCore.Years]
                                                                  <= self.year_end]
        self.energy_coal_variable_demand_df = self.energy_coal_variable_demand_df.loc[self.energy_coal_variable_demand_df[GlossaryCore.Years]
                                                                    >= self.year_start]
        self.energy_coal_variable_demand_df= self.energy_coal_variable_demand_df.loc[self.energy_coal_variable_demand_df[GlossaryCore.Years]
                                                                  <= self.year_end]
    def test_coal_resource_analytic_grad(self):
        
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        self.model_name = 'coal_resource'

        ns_dict = {'ns_public': f'{self.name}',
                   GlossaryCore.NS_WITNESS: f'{self.name}.{self.model_name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}.{self.model_name}',
                   'ns_coal_resource': f'{self.name}.{self.model_name}',
                   'ns_resource': f'{self.name}.{self.model_name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.core.core_resources.models.coal_resource.coal_resource_disc.CoalResourceDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{self.model_name}.resources_demand': self.energy_coal_demand_df
                       }
        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_coal_discipline.pkl',
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',local_data = disc_techno.local_data,
                            inputs=[
                                f'{self.name}.{self.model_name}.resources_demand'],
                            outputs=[f'{self.name}.{self.model_name}.resource_stock',
                                     f'{self.name}.{self.model_name}.resource_price',
                                     f'{self.name}.{self.model_name}.use_stock',
                                     f'{self.name}.{self.model_name}.predictable_production',
                                     f'{self.name}.{self.model_name}.recycled_production'
                                     ])
        
    def test_coal_resource_damand_variable_analytic_grad(self):
        
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        self.model_name = 'coal_resource'

        ns_dict = {'ns_public': f'{self.name}',
                   GlossaryCore.NS_WITNESS: f'{self.name}.{self.model_name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}.{self.model_name}',
                   'ns_coal_resource':f'{self.name}.{self.model_name}',
                   'ns_resource': f'{self.name}.{self.model_name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.core.core_resources.models.coal_resource.coal_resource_disc.CoalResourceDiscipline'
        builder = self.ee.factory.get_builder_from_module(self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        inputs_dict = {f'{self.name}.{GlossaryCore.YearStart}': self.year_start,
                       f'{self.name}.{GlossaryCore.YearEnd}': self.year_end,
                       f'{self.name}.{self.model_name}.resources_demand': self.energy_coal_variable_demand_df
                       }
        self.ee.load_study_from_input_dict(inputs_dict)

        self.ee.execute()

        disc_techno = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline

        self.check_jacobian(location=dirname(__file__), filename='jacobian_coal_demand_variable_discipline.pkl', 
                            discipline=disc_techno, step=1e-15, derr_approx='complex_step',local_data = disc_techno.local_data,
                            inputs=[
                                f'{self.name}.{self.model_name}.resources_demand'],
                            outputs=[f'{self.name}.{self.model_name}.resource_stock',
                                     f'{self.name}.{self.model_name}.resource_price',
                                     f'{self.name}.{self.model_name}.use_stock',
                                     f'{self.name}.{self.model_name}.predictable_production',
                                     f'{self.name}.{self.model_name}.recycled_production',
                                     ])