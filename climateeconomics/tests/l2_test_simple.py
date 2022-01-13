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

from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class SimpleTest(AbstractJacobianUnittest):
    '''    
    Very simple (and quick) test to setup the jenkins jobs for l2 test 
    and the launch at stable merge
    '''

    def setUp(self):

        self.name = 'Test'

    def analytic_grad_entry(self):
        return [
            self.test_01_simple
        ]

    def test_01_simple(self):

        self.name = 'Test'
        A = 2 + 2
        B = 4
        self.assertEqual(A, B, '2+2 != 4')
        print('Properly went through the test')
