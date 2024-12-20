"""
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
"""

import unittest

import numpy as np
import pandas as pd

from climateeconomics.core.tools.differentiable_model import DifferentiableModel


class ComplexTestModel(DifferentiableModel):
    def compute(self):
        x = self.inputs["x"]
        y = self.inputs["y"]
        z = self.inputs["z"]
        matrix = self.inputs["matrix"]

        self.outputs["output1"] = self.np.sin(x) * self.np.cos(y) + z**2
        self.outputs["output2"] = self.np.dot(matrix, self.np.array([x, y, z]))
        self.outputs["output3"] = {"a": x * y, "b": y * z, "c": x * z}


class TestComplexDifferentiableModel(unittest.TestCase):
    def setUp(self):
        self.model = ComplexTestModel()
        self.inputs = {
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
            "matrix": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        }
        self.model.set_inputs(self.inputs)

    def test_set_inputs(self):
        for key, value in self.inputs.items():
            np.testing.assert_array_equal(self.model.inputs[key], value)

    def test_set_inputs_with_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        self.model.set_inputs({"df": df})
        np.testing.assert_array_equal(self.model.inputs["df:a"], np.array([1, 2, 3]))
        np.testing.assert_array_equal(self.model.inputs["df:b"], np.array([4, 5, 6]))

    def test_compute(self):
        self.model.compute()
        x, y, z = self.inputs["x"], self.inputs["y"], self.inputs["z"]
        matrix = self.inputs["matrix"]

        expected_output1 = np.sin(x) * np.cos(y) + z**2
        expected_output2 = np.dot(matrix, np.array([x, y, z]))
        expected_output3 = {"a": x * y, "b": y * z, "c": x * z}

        np.testing.assert_almost_equal(self.model.outputs["output1"], expected_output1)
        np.testing.assert_array_almost_equal(
            self.model.outputs["output2"], expected_output2
        )
        for key, val in expected_output3.items():
            np.testing.assert_almost_equal(
                self.model.outputs["output3"][key],
                val,
            )

    def test_compute_partial(self):
        self.model.compute()
        partial_x = self.model.compute_partial("output1", "x")
        partial_y = self.model.compute_partial("output1", "y")
        partial_z = self.model.compute_partial("output1", "z")

        x, y, z = self.inputs["x"], self.inputs["y"], self.inputs["z"]
        expected_partial_x = np.cos(x) * np.cos(y)
        expected_partial_y = -np.sin(x) * np.sin(y)
        expected_partial_z = 2 * z

        np.testing.assert_almost_equal(partial_x, expected_partial_x)
        np.testing.assert_almost_equal(partial_y, expected_partial_y)
        np.testing.assert_almost_equal(partial_z, expected_partial_z)

    def test_compute_partial_all_inputs(self):
        self.model.compute()
        partials = self.model.compute_partial_all_inputs("output2")

        expected_partial_x = self.inputs["matrix"][:, 0]
        expected_partial_y = self.inputs["matrix"][:, 1]
        expected_partial_z = self.inputs["matrix"][:, 2]

        np.testing.assert_array_almost_equal(partials["x"], expected_partial_x)
        np.testing.assert_array_almost_equal(partials["y"], expected_partial_y)
        np.testing.assert_array_almost_equal(partials["z"], expected_partial_z)

    def test_set_parameters(self):
        params = {"a": 1.0, "b": np.array([1, 2, 3])}
        self.model.set_parameters(params)
        self.assertEqual(self.model.get_parameters(), params)

    def test_get_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        self.model.set_inputs({"df": df})
        self.model.outputs = {"df:a": np.array([1, 2, 3]), "df:b": np.array([4, 5, 6])}
        result_df = self.model.get_dataframe("df")
        pd.testing.assert_frame_equal(result_df, df, check_dtype=False)

    def test_get_dataframes(self):
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]})
        self.model.set_inputs({"df1": df1, "df2": df2})
        self.model.outputs = {
            "df1:a": np.array([1, 2, 3]),
            "df1:b": np.array([4, 5, 6]),
            "df2:c": np.array([7, 8, 9]),
            "df2:d": np.array([10, 11, 12]),
        }
        result_dfs = self.model.get_dataframes()
        pd.testing.assert_frame_equal(result_dfs["df1"], df1, check_dtype=False)
        pd.testing.assert_frame_equal(result_dfs["df2"], df2, check_dtype=False)

    def test_compute_partial_multiple(self):
        self.model.compute()
        partials = self.model.compute_partial("output1", ["x", "y", "z"])

        x, y, z = self.inputs["x"], self.inputs["y"], self.inputs["z"]
        expected_partial_x = np.cos(x) * np.cos(y)
        expected_partial_y = -np.sin(x) * np.sin(y)
        expected_partial_z = 2 * z

        np.testing.assert_almost_equal(partials["x"], expected_partial_x)
        np.testing.assert_almost_equal(partials["y"], expected_partial_y)
        np.testing.assert_almost_equal(partials["z"], expected_partial_z)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
