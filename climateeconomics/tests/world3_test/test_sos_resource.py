'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/15-2023/11/02 Copyright 2023 Capgemini

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
from climateeconomics.core.core_world3.resource import Resource

import unittest

from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.execution_engine import ExecutionEngine


def intialize_pyworld3_resource_inputs():
    obj = Resource()
    data = {GlossaryCore.YearStart: 1900,
            GlossaryCore.YearEnd: 2100,
            GlossaryCore.TimeStep: 0.5,
            'pyear': 1975}
    obj.set_data(data)
    obj.init_resource_constants()
    obj.init_resource_variables()
    obj.init_exogenous_inputs()
    obj.set_resource_table_functions()
    obj.set_resource_table_functions()
    obj.set_resource_delay_functions()
    obj.run_resource()
    return (obj)

def create_resource_input(name):
    ref = intialize_pyworld3_resource_inputs()

    values_dict = {name + ".pop": ref.pop,
                   name + ".iopc": ref.iopc}

    return values_dict


class TestSoSResource(unittest.TestCase):
    """
    SoSDiscipline test class
    """
    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'Test'
        self.model_name = GlossaryCore.SectorAgriculture
        self.ee = ExecutionEngine(self.name)

    def test_01_instantiate_sosdiscipline(self):
        '''
        default initialisation test
        '''

        ns_dict = {'ns_data': f'{self.name}.{self.model_name}',
                   'ns_coupling': f'{self.name}.{self.model_name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        # Get discipline builder using path
        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_world3.resource_discipline.ResourceDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        # Set builder in factory and configure
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        # Set input values

        values_dict = create_resource_input(f'{self.name}.{self.model_name}')

        values_dict[f'{self.name}.{self.model_name}' + GlossaryCore.YearStart] = 1900
        values_dict[f'{self.name}.{self.model_name}' + GlossaryCore.YearEnd] = 2100
        values_dict[f'{self.name}.{self.model_name}' + GlossaryCore.TimeStep] = 0.5
        values_dict[f'{self.name}.{self.model_name}' + 'pyear'] = 1975

        # print(data_dir)


        # Configure process with input values
        self.ee.load_study_from_input_dict(values_dict)

        # Execute process
        self.ee.execute()