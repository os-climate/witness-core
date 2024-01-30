'''
Copyright 2023 Capgemini

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
import pprint
import unittest

from sostrades_core.tools.check_headers import check_headers


class Testheader(unittest.TestCase):
    """
    Check headers test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.pp = pprint.PrettyPrinter(indent=4, compact=True)
        self.ExtensionToIgnore = ["pkl", "png", "jpg", "csv", "md", "markdown", "avif", "json", "in", "gitignore", "cfg", "puml", "pdf", "txt", "ipynb", "zip", "rst"]
        #Add here the files to ignore       
        self.FilesToIgnore = [#"climateeconomics/sos_processes/iam/witness/economics_sector_process/__init__.py",
                              #"climateeconomics/sos_processes/iam/witness/climate_process/__init__.py",
                              "climateeconomics/sos_processes/iam/witness/witness_coarse_Ku_optim_process/process.py",
                              "climateeconomics/sos_processes/iam/witness/witness_coarse_story_telling_optim_process/process.py",
                              "default_process_rights.yaml"]
        #commit from where to compare added, modeified deleted ...
        self.airbus_rev_commit = "fb7c7e2e92dc37b1b6a7f8e968de806f981199a0"

    def test_Headers(self):
        check_headers(self.ExtensionToIgnore,self.FilesToIgnore,self.airbus_rev_commit)

        

