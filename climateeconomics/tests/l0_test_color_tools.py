'''
Copyright 2025 Capgemini

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

import doctest
import unittest

import climateeconomics.core.tools.color_map as color_map
import climateeconomics.core.tools.color_palette as color_palette
import climateeconomics.core.tools.color_tools as color_tools


class TestColorTools(unittest.TestCase):
    def test_doctests(self):
        # Run doctests on the module
        results = doctest.testmod(color_tools)
        # Assert that all tests passed
        self.assertEqual(results.failed, 0, f"{results.failed} doctests failed.")


class TestColorPalette(unittest.TestCase):
    def test_doctests(self):
        # Run doctests on the module
        results = doctest.testmod(color_palette)
        # Assert that all tests passed
        self.assertEqual(results.failed, 0, f"{results.failed} doctests failed.")


class TestColorMap(unittest.TestCase):
    def test_doctests(self):
        # Run doctests on the module
        results = doctest.testmod(color_map)
        # Assert that all tests passed
        self.assertEqual(results.failed, 0, f"{results.failed} doctests failed.")


if __name__ == '__main__':
    unittest.main()
