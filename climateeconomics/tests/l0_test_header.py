'''
Copyright 2024 Capgemini

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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
'''

import HeadersIgnoreConfig
from sostrades_core.tools.check_headers import HeaderTools
import unittest
import pprint

class Testheader(unittest.TestCase):
    """
    Check headers test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.pp = pprint.PrettyPrinter(indent=4, compact=True)
        self.extension_to_ignore = HeadersIgnoreConfig.extension_to_ignore
        #Add here the files to ignore  
        self.files_to_ignore = HeadersIgnoreConfig.files_to_ignore
        #commit from where to compare added, modeified deleted ...
        self.airbus_rev_commit = HeadersIgnoreConfig.airbus_rev_commit

    def test_Headers(self):
        ht = HeaderTools()
        ht.check_headers(self.extension_to_ignore, self.files_to_ignore, self.airbus_rev_commit)
