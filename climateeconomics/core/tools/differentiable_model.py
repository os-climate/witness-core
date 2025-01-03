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

import functools  # do not remove this grey imports otherwise pylint fails
import time  # do not remove this grey imports otherwise pylint fails
from contextlib import ContextDecorator, contextmanager
from copy import deepcopy
from statistics import mean
from typing import Any, Callable, Union

import autograd.numpy as np
import numpy.typing as npt
import pandas as pd
from autograd import grad, jacobian
from autograd.builtins import dict, isinstance
from tqdm import tqdm

from climateeconomics.glossarycore import GlossaryCore

ArrayLike = Union[list[float], npt.NDArray[np.float64]]
InputType = Union[float, int, ArrayLike, pd.DataFrame]
OutputType = Union[float, ArrayLike]


class TimerContext(ContextDecorator):
    def __init__(self, name: str = "Code block", runs: int = 5):
        self.name = name
        self.runs = runs
        self.execution_times = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        avg_time = mean(self.execution_times)
        max_time = max(self.execution_times)
        min_time = min(self.execution_times)

        print(f"'{self.name}' statistics:")
        print(f"  Average time: {avg_time:.6f} seconds")
        print(f"  Maximum time: {max_time:.6f} seconds")
        print(f"  Minimum time: {min_time:.6f} seconds")
        print(f"  Number of runs: {self.runs}")

    def run(self, func, *args, **kwargs):
        for _ in tqdm(range(self.runs)):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            self.execution_times.append(end_time - start_time)
        return result

timer = TimerContext


def time_function(runs: int = 5) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            execution_times = []
            results = []

            for _ in range(runs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                results.append(result)

            avg_time = mean(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)

            print(f"Function '{func.__name__}' statistics:")
            print(f"  Average time: {avg_time:.6f} seconds")
            print(f"  Maximum time: {max_time:.6f} seconds")
            print(f"  Minimum time: {min_time:.6f} seconds")
            print(f"  Number of runs: {runs}")

            return results[0]  # Return the result of the first run

        return wrapper
    return decorator

@contextmanager
def timer_context(name: str = "Operation"):
    """
    Context manager for timing a block of code.
    
    Args:
        name: Description of the operation being timed
        
    Yields:
        None
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"{name} took {execution_time:.4f} seconds to execute")


class DifferentiableModel:
    """A base class for differentiable models.

    This class provides a framework for creating models that can be differentiated
    with respect to their inputs. It handles input setting, output computation,
    and gradient/Jacobian calculations.

    Attributes:
        inputs (dict): A dictionary to store input values.
        outputs (dict): A dictionary to store computed output values.
        parameters (dict): A dictionary to store model parameters.
        output_types (dict): A dictionary to store output types.
    """

    def __init__(self, flatten_dfs: bool = True):
        """
        Initialize the model.

        Args:
            flatten_dfs: If True, DataFrames will be flattened into separate arrays
                        with keys as 'dataframe_name:column_name'.
                        If False, DataFrames will be converted to dictionaries of arrays.
        """
        self.dataframes_outputs_colnames: dict[str: list[str]] = {}
        self.dataframes_inputs_colnames: dict[str: list[str]] = {}
        self.inputs: dict[str, Union[float, np.ndarray, dict[str, np.ndarray]]] = {}
        self.outputs: dict[str, Union[float, np.ndarray, dict[str, np.ndarray]]] = {}
        self.flatten_dfs = flatten_dfs

    def set_parameters(self, params: dict[str, float, np.ndarray]) -> None:
        """Sets the parameters of the model.

        Args:
            params (dict): A dictionary of parameter names and their values.
        """
        self.parameters = params

    def get_parameters(self) -> dict[str, float, np.ndarray]:
        """Retrieves the current parameters of the model.

        Returns:
            dict: A dictionary of parameter names and their values.
        """
        return self.parameters

    def set_inputs(self, inputs: dict[str, InputType]) -> None:
        """Sets the input values for the model.

        Args:
            inputs (dict): A dictionary containing input names and their values.

        Raises:
            TypeError: If a DataFrame input contains non-numeric data.
            ValueError: If an input array has more than 2 dimensions.
        """

        self.dataframes_inputs_colnames = {}
        for key, value in inputs.items():
            if isinstance(value, pd.DataFrame):
                if not all(np.issubdtype(dtype, np.number) for dtype in value.dtypes):
                    msg = f"DataFrame '{key}' contains non-numeric data, which is not supported."
                    raise TypeError(msg)
                if self.flatten_dfs:
                    self.dataframes_inputs_colnames[key] = list(value.columns)
                    for col in value.columns:
                        self.inputs[f"{key}:{col}"] = value[col].to_numpy()
                else:
                    self.inputs[key] = {
                        col: value[col].to_numpy() for col in value.columns
                    }
            elif isinstance(value, pd.Series):
                self.inputs[key] = value.to_numpy()

            elif isinstance(value, (list, np.ndarray)):
                if len(np.array(value).shape) > 2:
                    msg = f"Input '{key}' has too many dimensions; only 1D or 2D arrays are allowed."
                    raise ValueError(msg)
                self.inputs[key] = np.array(value)
            else:
                self.inputs[key] = value

    def set_output_types(self, output_types: dict[str, str]) -> None:
        """Sets the types of the output variables.

        Args:
            output_types (dict): A dictionary of output names and their types.
        """
        self.output_types = output_types

    def get_dataframe(self, name: str):
        """
        Retrieve a specific DataFrame from outputs based on its name.
        Works with both dictionary outputs and flattened outputs.

        Args:
            name: Name of the DataFrame to retrieve

        Returns:
            DataFrame if it can be constructed from outputs, None otherwise
        """
        # First check if there's a direct dictionary output with this name
        if name in self.outputs and isinstance(self.outputs[name], dict):
            # Check if all values are arrays
            if all(isinstance(v, np.ndarray) for v in self.outputs[name].values()):
                return pd.DataFrame(self.outputs[name])

        # If using flatten_dfs, check for columns with this base name
        if self.flatten_dfs:
            prefix = f"{name}:"
            columns = {}
            for key, value in self.outputs.items():
                if key.startswith(prefix) and isinstance(value, np.ndarray):
                    col_name = key[
                        len(prefix) :
                    ]  # Remove the prefix to get column name
                    columns[col_name] = value

            if columns:  # Only create DataFrame if we found matching columns
                return pd.DataFrame(columns)

        return None

    def get_dataframes(self) -> dict[str, pd.DataFrame]:
        """
        Convert all suitable outputs to pandas DataFrames.

        Returns:
            Dictionary of DataFrames reconstructed from outputs
        """
        result = {}
        self.dataframes_outputs_colnames = {}
        if self.flatten_dfs:
            # Find all unique base names in flattened outputs
            base_names = set(
                key.split(":", 1)[0] for key in self.outputs.keys() if ":" in key
            )
            for base_name in base_names:
                df = self.get_dataframe(base_name)
                if df is not None:
                    result[base_name] = df
                    self.dataframes_outputs_colnames[base_name] = list(df.columns)

        # Check for dictionary outputs
        for key, value in self.outputs.items():
            if isinstance(value, dict):
                df = self.get_dataframe(key)
                if df is not None:
                    result[key] = df

        return result

    def compute(self, *args: InputType) -> OutputType:
        """Computes the model outputs based on inputs passed as arguments.
        This method should be overridden by subclasses.

        Args:
            *args: Variable length argument list of input values.

        Returns:
            OutputType: The computed output.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        msg = "Subclasses must implement the compute method."
        raise NotImplementedError(msg)

    def get_outputs(self) -> dict[str, OutputType]:
        """Retrieves the computed outputs.

        Returns:
            dict: A dictionary of output names and their computed values.
        """
        return self.outputs

    def _wrap_compute_for_autograd(self, output_name: str, input_name: str) -> tuple[Callable, bool, list[str]]:
        """
        Wraps the compute method to work with autograd for a single input.
        For dictionary inputs, creates a wrapper that takes individual arguments for each key.

        Args:
            output_name: Name of the output to differentiate
            input_name: Name of the input to differentiate with respect to

        Returns:
            Tuple of:
            - Wrapped compute function
            - Boolean indicating if input is dictionary
            - List of input component names (for dictionaries) or [input_name] (for non-dictionaries)
        """
        original_input = self.inputs[input_name]
        is_dict_input = isinstance(original_input, dict)

        if is_dict_input:
            input_components = list(original_input.keys())

            def wrapped_compute(*args):
                # Store original inputs
                original_inputs = deepcopy(self.inputs)

                try:
                    # Update inputs with the arguments
                    temp_inputs = deepcopy(original_inputs)

                    # Create dictionary from individual arguments
                    temp_inputs[input_name] = {
                        key: np.array(value)
                        for key, value in zip(input_components, args)
                    }

                    # Set temporary inputs
                    self.inputs = temp_inputs

                    # Compute
                    self.compute()

                    # Get the output we want to differentiate
                    result = self.outputs[output_name]

                    if isinstance(result, dict):
                        result = {k: np.array(v) for k, v in result.items()}
                    else:
                        result = np.array(result)

                    return result

                finally:
                    # Restore original inputs
                    self.inputs = original_inputs

            return wrapped_compute, True, input_components

        else:
            def wrapped_compute(x):
                # Store original inputs
                original_inputs = deepcopy(self.inputs)

                try:
                    # Update inputs with the argument
                    temp_inputs = deepcopy(original_inputs)
                    temp_inputs[input_name] = np.array(x)

                    # Set temporary inputs
                    self.inputs = temp_inputs

                    # Compute
                    self.compute()

                    # Get the output we want to differentiate
                    result = self.outputs[output_name]

                    if isinstance(result, dict):
                        result = {k: np.array(v) for k, v in result.items()}
                    else:
                        result = np.array(result)

                    return result

                finally:
                    # Restore original inputs
                    self.inputs = original_inputs

            return wrapped_compute, False, [input_name]

    def compute_partial_2(self, output_name: str, input_names: list[str]) -> dict[str, Any]:
        """
        Compute partial derivatives of specified output with respect to specified inputs.
        Uses Jacobian for array outputs and gradient for scalar outputs.

        Args:
            output_name: Name of the output to differentiate
            input_names: List of input names to differentiate with respect to

        Returns:
            Dictionary mapping input names to their gradients/jacobians
        """
        # pylint: disable=E1120

        result = {}

        # Check if output is scalar or array
        test_output = self.outputs[output_name]
        is_scalar = np.isscalar(test_output) if not isinstance(test_output, dict) else False

        for input_name in input_names:
            wrapped_function, is_dict_input, input_components = self._wrap_compute_for_autograd(
                output_name, input_name
            )

            # Use appropriate differentiation function
            if is_scalar:
                # For scalar outputs, use grad
                diff_function = grad(wrapped_function)
            else:
                # For array outputs, use jacobian
                diff_function = jacobian(wrapped_function)

            # Prepare input values for gradient computation
            if is_dict_input:
                input_values = [np.array(self.inputs[input_name][k]) for k in input_components]
                # Compute gradient/jacobian for each component
                grads = diff_function(*input_values)

                # Package results into dictionary
                if isinstance(grads, tuple):
                    result[input_name] = {
                        component: grad for component, grad in zip(input_components, grads)
                    }
                else:
                    # Handle case where there's only one component
                    result[input_name] = {input_components[0]: grads}
            else:
                input_value = np.array(self.inputs[input_name])
                result[input_name] = diff_function(input_value)

        return result

    def _inputs_to_array(self, keys):
        """
        Convert selected inputs items into a 1D numpy array.

        Args:
            keys (list): List of keys to include in the array

        Returns:
            tuple: (concatenated array, list of shapes, total length)
        """
        arrays = []
        shapes = []
        total_length = 0

        for key in keys:
            if isinstance(self.inputs[key], float):
                arr = np.array([self.inputs[key]])
                shapes.append(arr.shape)
            else:
                arr = self.inputs[key].reshape(-1)  # Flatten the array
                shapes.append(self.inputs[key].shape)

            total_length += len(arr)
            arrays.append(arr)

        return np.concatenate(arrays), shapes, total_length

    def _array_to_dict(self, array, keys, shapes):
        """
        Convert 1D array back to dictionary with original shapes.

        Args:
            array (np.ndarray): 1D array containing all values
            keys (list): List of keys in the same order as dict_to_array
            shapes (list): Original shapes of arrays from dict_to_array

        Returns:
            dict: Dictionary with reshaped arrays
        """
        result = {}
        start_idx = 0

        for key, shape in zip(keys, shapes):
            size = np.prod(shape)
            arr = array[start_idx : start_idx + size]
            result[key] = arr.reshape(shape)
            start_idx += size

        return result

    def _create_wrapped_compute_array(
        self, output_name: str, input_names: list[str] = None
    ) -> Callable:
        """Creates a wrapped compute function that accepts a single array for multiple inputs.

        Args:
            output_name (str): The name of the output.
            input_names (List[str]): List of input names to include in the wrapper.
                If None, all inputs are used.

        Returns:
            Callable: A wrapped compute function that accepts a single 1D numpy array.
        """
        if input_names is None:
            input_names = list(self.inputs.keys())

        # Get the shapes once to avoid repeated calls
        _, shapes, _ = self._inputs_to_array(input_names)

        def wrapped_compute(flat_array: np.ndarray):
            # Store original state
            temp_inputs = deepcopy(self.inputs)
            #temp_outputs = deepcopy(self.outputs)

            # Convert flat array back to dictionary and update inputs
            restored_dict = self._array_to_dict(flat_array, input_names, shapes)
            for key, value in restored_dict.items():
                self.inputs[key] = value

            # Compute and get result
            self.compute()
            return_value = self.outputs[output_name]

            # Restore original state
            self.inputs = temp_inputs
            #self.outputs = temp_outputs

            return return_value

        return wrapped_compute

    def _create_wrapped_compute(self, output_name: str, input_name: str) -> Callable:
        """Creates a wrapped compute function for a specific output and input.

        Args:
            output_name (str): The name of the output.
            input_name (str): The name of the input. If empty, all inputs are used.

        Returns:
            Callable: A wrapped compute function.
        """

        
        # Create a function that handles a specific input
        if isinstance(self.inputs[input_name], dict):
            def wrapped_compute(*args):
                temp_inputs = deepcopy(self.inputs)
                for i, col in enumerate(self.inputs[input_name].keys()):
                    self.inputs[input_name][col] = args[i]
                self.compute()
                self.inputs = temp_inputs
                return self.outputs[output_name]
        else:

            def wrapped_compute(arg):
                temp_inputs = deepcopy(self.inputs)
                self.inputs[input_name] = arg
                self.compute()
                self.inputs = temp_inputs
                return self.outputs[output_name]

        return wrapped_compute

    def compute_partial(
        self, output_name: str, input_names: Union[str, list]
    ) -> Union[
        npt.NDArray[np.float64],
        dict[str, Union[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]],
    ]:
        """Computes the partial derivative of an output with respect to an input or all inputs.

        Args:
            output_name (str): The name of the output.
            input_names (str): The name of the input.

        Returns:
            Union[npt.NDArray[np.float64], Dict[str, Union[npt.NDArray[np.float64], Dict[str, npt.NDArray[np.float64]]]]]:
                The computed partial derivative(s).
        """
        # pylint: disable=E1120

        # go to compute_partial_2 if given a list of input_names
        # if isinstance(input_names, list):
        #     return self.compute_partial_2(output_name, input_names)

        is_single = False
        if isinstance(input_names, str):
            is_single = True
            input_names = [input_names]

        result = {}
        for input_name in input_names:
            wrapped_compute = self._create_wrapped_compute(output_name, input_name)

            if isinstance(self.inputs[input_name], dict):  # For DataFrame inputs
                jacobians = {}
                for col in self.inputs[input_name]:
                    jacobian_func = jacobian(
                        wrapped_compute,
                        argnum=list(self.inputs[input_name].keys()).index(col),
                    )
                    jacobians[col] = jacobian_func(*self.inputs[input_name].values())
                return jacobians

            jacobian_func = jacobian(wrapped_compute)
            result[input_name] = jacobian_func(self.inputs[input_name])

        if is_single:
            return result[input_names[0]]

        return result

    def compute_partial_multiple(
        self, output_name: str, input_names: Union[str, list]
    ) -> Union[
        npt.NDArray[np.float64],
        dict[str, Union[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]],
    ]:
        """Computes the partial derivative of an output with respect to an input or all inputs.

        Args:
            output_name (str): The name of the output.
            input_names (str): The name of the input.

        Returns:
            Union[npt.NDArray[np.float64], Dict[str, Union[npt.NDArray[np.float64], Dict[str, npt.NDArray[np.float64]]]]]:
                The computed partial derivative(s).
        """
        # pylint: disable=E1120

        wrapped_compute = self._create_wrapped_compute_array(output_name, input_names)

        inputs_array, shapes, _ = self._inputs_to_array(input_names)

        jacobian_func = jacobian(wrapped_compute)
        jac_array = jacobian_func(inputs_array)
        
        # Convert Jacobian array back to dictionary format
        output_shape = self.outputs[output_name].shape
        result = {}
        start_idx = 0
        
        for key, shape in zip(input_names, shapes):
            size = np.prod(shape)
            # Reshape the Jacobian slice for this input
            # Combine output shape with input shape
            full_shape = output_shape + shape
            jac_slice = jac_array[:, start_idx:start_idx + size].reshape(full_shape)
            result[key] = jac_slice
            start_idx += size

        return result

    def compute_partial_all_inputs(self, output_name: str):
        # Compute gradients for all inputs
        result = {}
        for key in self.inputs:
            partial = self.compute_partial(output_name, key)
            result[key] = partial
        return result
        # return self.compute_partial_2(output_name, self.inputs.keys())

    def check_partial(
        self,
        output_name: str,
        input_name: str,
        method: str = "complex_step",
        epsilon: float = 1e-8,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> dict[str, Union[np.ndarray, float, bool]]:
        """
        Compares the partial derivative computed by compute_partial with a numerical approximation
        for a specific input-output pair, handling array inputs correctly.

        Args:
            output_name (str): The name of the output.
            input_name (str): The name of the input.
            method (Literal["finite_differences", "complex_step"]): The numerical method to use.
            epsilon (float): Step size for numerical approximation.
            rtol (float): Relative tolerance for comparison.
            atol (float): Absolute tolerance for comparison.

        Returns:
            Dict[str, Union[np.ndarray, float, bool]]: A dictionary containing the analytical derivative,
            numerical approximation, maximum absolute error, maximum relative error, and whether the
            results are within tolerance.
        """

        def finite_difference(f, x, i, eps=epsilon):
            x_plus = x.copy()
            x_plus[i] += eps
            return (f(x_plus) - f(x)) / eps

        def complex_step(f, x, i, eps=epsilon):
            x_complex = x.copy().astype(complex)
            x_complex[i] += 1j * eps
            return np.imag(f(x_complex)) / eps

        # Get the analytical partial derivative
        analytical = self.compute_partial(output_name, input_name)

        # Prepare for numerical approximation
        wrapped_compute = self._create_wrapped_compute(output_name, input_name)

        if isinstance(self.inputs[input_name], dict):
            numerical = {}
            for col, value in self.inputs[input_name].items():
                if np.isscalar(value):

                    def f(x):
                        temp_inputs = self.inputs[input_name].copy()
                        temp_inputs[col] = x
                        return wrapped_compute(*temp_inputs.values())

                    if method == "finite_differences":
                        numerical[col] = finite_difference(f, np.array([value]), 0)
                    else:  # complex_step
                        numerical[col] = complex_step(f, np.array([value]), 0)
                else:  # array input
                    numerical[col] = np.zeros(
                        value.shape + self.outputs[output_name].shape
                    )
                    for i in np.ndindex(value.shape):

                        def f(x):
                            temp_inputs = self.inputs[input_name].copy()
                            temp_inputs[col] = x
                            return wrapped_compute(*temp_inputs.values())

                        if method == "finite_differences":
                            numerical[col][i] = finite_difference(f, value, i)
                        else:  # complex_step
                            numerical[col][i] = complex_step(f, value, i)
        else:
            value = self.inputs[input_name]
            if np.isscalar(value):
                if method == "finite_differences":
                    numerical = finite_difference(wrapped_compute, np.array([value]), 0)
                else:  # complex_step
                    numerical = complex_step(wrapped_compute, np.array([value]), 0)
            else:  # array input
                numerical = np.zeros(value.shape + self.outputs[output_name].shape)
                for i in np.ndindex(value.shape):
                    if method == "finite_differences":
                        numerical[i] = finite_difference(wrapped_compute, value, i)
                    else:  # complex_step
                        numerical[i] = complex_step(wrapped_compute, value, i)

        # Ensure analytical and numerical have the same shape
        if isinstance(analytical, dict):
            for col in analytical:
                analytical[col] = np.atleast_1d(analytical[col])
                numerical[col] = np.atleast_1d(numerical[col])
        else:
            analytical = np.atleast_1d(analytical)
            numerical = np.atleast_1d(numerical)

        if isinstance(numerical, np.ndarray):
            numerical = numerical.T
        elif isinstance(numerical, dict):
            numerical = {k: v.T for k, v in numerical.items()}

        # Compute errors
        if isinstance(analytical, dict):
            abs_error = {
                col: np.abs(analytical[col] - numerical[col]) for col in analytical
            }
            rel_error = {
                col: abs_error[col] / (np.abs(analytical[col]) + 1e-15)
                for col in analytical
            }
            max_abs_error = max(np.max(err) for err in abs_error.values())
            max_rel_error = max(np.max(err) for err in rel_error.values())
            within_tolerance = all(
                np.allclose(analytical[col], numerical[col], rtol=rtol, atol=atol)
                for col in analytical
            )
        else:
            abs_error = np.abs(analytical - numerical)
            rel_error = abs_error / (np.abs(analytical) + 1e-15)
            max_abs_error = np.max(abs_error)
            max_rel_error = np.max(rel_error)
            within_tolerance = np.allclose(analytical, numerical, rtol=rtol, atol=atol)

        return {
            "analytical": analytical,
            "numerical": numerical,
            "max_absolute_error": float(max_abs_error),
            "max_relative_error": float(max_rel_error),
            "within_tolerance": within_tolerance,
        }

    def compute_jacobians_custom(self, outputs: list[str], inputs: list[str]) -> dict[str: dict[str: dict[str: dict[str: np.ndarray]]]]:
        """
        Returns a dictionnary 'gradients' containing gradients for SoSwrapp disciplines, with structure :
        gradients[output df name][output column name][input df name][input column name] = value
        """
        gradients = {}
        all_inputs_paths = []
        for input_df_name in inputs:
            all_inputs_paths.extend(self.get_df_input_dotpaths(input_df_name))
        all_inputs_paths = list(filter(lambda x: not (str(x).endswith(f':{GlossaryCore.Years}')), all_inputs_paths))
        for i, output in enumerate(outputs):
            gradients[output] = {}
            output_columns_paths = list(filter(lambda x: not (str(x).endswith(f':{GlossaryCore.Years}')), self.get_df_output_dotpaths(output)))
            for output_path in output_columns_paths:
                gradients_output_path = self.compute_partial_multiple(output_name=output_path, input_names=all_inputs_paths)
                output_colname = output_path.split(f'{output}:')[1]
                gradients[output][output_colname] = {}
                for ip, value_grad in gradients_output_path.items():
                    input_varname, input_varname_colname = ip.split(':')
                    if input_varname in gradients[output][output_colname]:
                        gradients[output][output_colname][input_varname][input_varname_colname] = value_grad
                    else:
                        gradients[output][output_colname][input_varname] = {input_varname_colname: value_grad}

        return gradients
    def get_df_input_dotpaths(self, df_inputname: str) -> dict[str: list[str]]:
        return [f'{df_inputname}:{colname}' for colname in self.dataframes_inputs_colnames[df_inputname]]

    def get_df_output_dotpaths(self, df_outputname: str) -> dict[str: list[str]]:
        return [f'{df_outputname}:{colname}' for colname in self.dataframes_outputs_colnames[df_outputname]]


if __name__ == "__main__":

    import functools
    import time
    from contextlib import contextmanager
    from typing import Any, Callable


    class MyModel(DifferentiableModel):
        def compute(self) -> None:
            x = self.inputs["x"]
            y = self.inputs["y"]

            y_a = np.array(y["a"]) ** 3
            y_b = np.array(y["b"]) ** 3

            result = np.sum(x**2)
            result = result + np.sum(y_a)
            result = result + np.sum(y_b**3)

            self.outputs["result"] = np.array([result])

    # Usage example
    model = MyModel(flatten_dfs=False)

    # Set inputs
    inputs: dict[str, InputType] = {
        "x": np.array([1.0, 2.0, 3.0]),
        "y": pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}),
    }
    model.set_inputs(inputs)

    # Compute the model
    model.compute()

    # Get outputs
    outputs: dict[str, OutputType] = model.get_outputs()

    # Compute Jacobian
    jacobian_x = model.compute_partial("result", ["x", "y"])
    print("Jacobian_x:", jacobian_x)

    jacobian_y = model.compute_partial("result", "y")
    print("Jacobian_y:", jacobian_y)

    jacobian_y = model.compute_partial_all_inputs("result")
    print("Jacobian all:", jacobian_y)

    result = model.check_partial("result", "y", method="complex_step")
    print(f"Analytical: {result['analytical']}")
    print(f"Numerical: {result['numerical']}")
    print(f"Max absolute error: {result['max_absolute_error']}")
    print(f"Max relative error: {result['max_relative_error']}")
    print(f"Within tolerance: {result['within_tolerance']}")

    # Example with flatten_dfs=True
    class FlatModule(DifferentiableModel):
        def compute(self):
            x = self.inputs["data:feature1"]
            y = self.inputs["data:feature2"]

            # Create flattened outputs
            self.outputs["result:squared"] = x ** 2
            self.outputs["result:sum"] = x + y
            self.outputs["other:value"] = x * y

    model = FlatModule(flatten_dfs=True)
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    model.set_inputs({"data": df})
    model.compute()

    with timer_context("MULTIPLE"):
        for i in range(100):
            jacobian_y = model.compute_partial_multiple(
                "other:value", ["data:feature1", "data:feature2"]
            )
        print("Jacobian multiple:", jacobian_y)

    with timer_context("SINGLE ALL"):
        for i in range(100):
            jacobian_y = model.compute_partial_all_inputs("other:value")
        print("Jacobian multiple:", jacobian_y)

    for o in model.outputs:
        jacobian_y = model.compute_partial_all_inputs(o)
        print(f"Jacobian ({o}):", jacobian_y)

    # # Get a specific DataFrame
    # result_df = model.get_dataframe('result')  # DataFrame with 'squared' and 'sum' columns
    # other_df = model.get_dataframe('other')   # DataFrame with 'value' column

    # # Get all DataFrames
    # all_dfs = model.get_dataframes()  # Dictionary with 'result' and 'other' keys

    # Example with flatten_dfs=False
    class DictModule(DifferentiableModel):
        def compute(self):
            x = self.inputs["data"]["feature1"]
            y = self.inputs["data"]["feature2"]

            # Create dictionary outputs
            self.outputs["result:squared"] = x ** 2
            self.outputs["result:sum"] = x + y
            self.outputs["single_value"] = (
                x.mean()
            )  # This won't be converted to DataFrame

    model = DictModule(flatten_dfs=False)
    model.set_inputs({"data": df})
    model.compute()

    for o in model.outputs:
        j = model.compute_partial_all_inputs(o)
        print(f"Jacobian ({o}):", j)

    j = model.compute_partial("result:sum", "data")
    print("Jacobian (result:sum):", j)

    # # Get a specific DataFrame
    # result_df = model.get_dataframe('result')  # DataFrame with 'squared' and 'sum' columns
    # single_value_df = model.get_dataframe('single_value')  # Returns None

    # # Get all DataFrames
    # all_dfs = model.get_dataframes()  # Dictionary with only 'result' key
