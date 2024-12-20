from __future__ import annotations

import functools
import time
from contextlib import ContextDecorator, contextmanager
from copy import deepcopy
from statistics import mean
from typing import Any, Callable, Union

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

import autograd
import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

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
    """Decorate a function to measure the execution time.

    The function is run multiple times and the statistics of the
    execution times are printed to the console.

    Parameters
    ----------
    runs : int
        The number of times to run the function. Defaults to 5.

    Returns
    -------
    A decorator that runs the function multiple times and prints the execution
    time statistics.

    Example
    -------
    @time_function()
    def my_function(x):
        # Do something
        return x

    my_function(5)
    """

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
    """Context manager for timing a block of code.

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

    def __init__(
            self,
            flatten_dfs: bool = True,
            ad_backend: str = "autograd",
            overload_numpy: bool = True,
            numpy_ns: str = "np",
    ) -> None:
        """Initialize the model.

        Args:
            flatten_dfs: If True, DataFrames will be flattened into separate arrays
                        with keys as 'dataframe_name:column_name'.
                        If False, DataFrames will be converted to dictionaries of arrays.
            ad_backend: The backend to use for automatic differentiation. Defaults to "autograd".
            overload_numpy: If True, the numpy namespace will be replaced with the
                            appropriate namespace for the backend. Defaults to True.
            numpy_ns: The namespace to use for numpy if overload_numpy is True. Defaults to 'np'.

        """
        if ad_backend == "jax" and not HAS_JAX:
            error_msg = "JAX not installed. Please install JAX to use JAX backend."
            raise ValueError(error_msg)

        self._ad_backend = ad_backend

        # Overload numpy namespace if required / requested
        if overload_numpy:
            self.numpy_ns = numpy_ns
            self.np = anp if self._ad_backend == "autograd" else jnp
        else:
            self.np = np

        # Prepare ad_backend functions
        if ad_backend == "autograd":
            self.__grad = autograd.grad
            self.__jacobian = autograd.jacobian
        elif ad_backend == "jax":
            self.__grad = jax.grad
            self.__jacobian = jax.jacobian

        self.inputs: dict[str, Union[float, np.ndarray, dict[str, np.ndarray]]] = {}
        self.outputs: dict[str, Union[float, np.ndarray, dict[str, np.ndarray]]] = {}

        self._params = {}
        self._output_types = {}

        self.flatten_dfs = flatten_dfs

        # Default methods
        self.compute_partial = self.compute_partial_bwd

    @property
    def parameters(self) -> dict[str, float, np.ndarray, dict]:
        """Get the current parameters of the model.

        Returns:
            dict: A dictionary of parameter names and their values.

        """
        return self.get_parameters()

    @parameters.setter
    def parameters(self, value: dict[str, float, np.ndarray, dict]) -> None:
        """Set the parameters of the model.

        Args:
            value (dict): A dictionary of parameter names and their values.

        """
        self.set_parameters(value)

    def set_parameters(self, params: dict[str, float, np.ndarray]) -> None:
        """Set the parameters of the model.

        Args:
            params (dict): A dictionary of parameter names and their values.

        """
        self._params = params

    def get_parameters(self) -> dict[str, float, np.ndarray]:
        """Retrieve the current parameters of the model.

        Returns:
            dict: A dictionary of parameter names and their values.

        """
        return self._params

    def set_inputs(self, inputs_in: dict[str, InputType]) -> None:
        """Set the input values for the model.

        Args:
            inputs_in (dict): A dictionary containing input names and their values.

        Raises:
            TypeError: If a DataFrame input contains non-numeric data.
            ValueError: If an input array has more than 2 dimensions.

        """
        inputs = {}

        for key, value in inputs_in.items():
            if isinstance(value, pd.DataFrame):
                if not all(np.issubdtype(dtype, np.number) for dtype in value.dtypes):
                    msg = f"DataFrame '{key}' contains non-numeric data."
                    raise TypeError(msg)
                if self.flatten_dfs:
                    for col in value.columns:
                        inputs[f"{key}:{col}"] = value[col].to_numpy()
                else:
                    inputs[key] = {col: value[col].to_numpy() for col in value.columns}
            elif isinstance(value, pd.Series):
                inputs[key] = value.to_numpy()

            elif isinstance(value, (list, np.ndarray)):
                if len(np.array(value).shape) > 2:
                    msg = f"Input '{key}' has too many dimensions; only 1D or 2D arrays allowed."
                    raise ValueError(msg)
                inputs[key] = np.array(value)
            else:
                inputs[key] = value

        self.inputs = inputs

    def set_output_types(self, output_types: dict[str, str]) -> None:
        """Set the types of the output variables.

        Args:
            output_types (dict): A dictionary of output names and their types.

        """
        self.output_types = output_types

    def get_dataframe(
            self,
            name: str,
            get_from: str = "outputs",
    ) -> pd.DataFrame | None:
        """Retrieve a specific DataFrame from outputs or inputs based on its name.

        Works with both dictionary outputs and flattened outputs.

        Args:
            name: Name of the DataFrame to retrieve
            get_from: Source of the DataFrame, either "outputs" or "inputs"

        Returns:
            DataFrame if it can be constructed from outputs or inputs, None otherwise

        """
        if get_from == "inputs":
            source = self.inputs
        elif get_from == "outputs":
            source = self.outputs
        else:
            source = self.outputs

        # First check if there's a direct dictionary output with this name
        if (
                name in source
                and isinstance(source[name], dict)
                and all(isinstance(v, np.ndarray) for v in source[name].values())
        ):
            return pd.DataFrame(source[name])

        # If using flatten_dfs, check for columns with this base name
        if self.flatten_dfs:
            prefix = f"{name}:"
            columns = {}
            for key, value in source.items():
                if key.startswith(prefix) and isinstance(value, np.ndarray):
                    col_name = key[
                               len(prefix):
                               ]  # Remove the prefix to get column name
                    columns[col_name] = value

            if columns:  # Only create DataFrame if we found matching columns
                return pd.DataFrame(columns)

        return None

    def get_dataframes(self, get_from: str = "outputs") -> dict[str, pd.DataFrame]:
        """Convert all suitable outputs or inputs to pandas DataFrames.

        Args:
            get_from: Source of the DataFrame, either "outputs" or "inputs".
                Defaults to "outputs".

        Returns:
            Dictionary of DataFrames reconstructed from outputs

        """
        result = {}

        if get_from == "inputs":
            source = self.inputs
        elif get_from == "outputs":
            source = self.outputs
        else:
            source = self.outputs

        if self.flatten_dfs:
            # Find all unique base names in flattened outputs
            base_names = {key.split(":", 1)[0] for key in source if ":" in key}
            for base_name in base_names:
                df = self.get_dataframe(base_name)
                if df is not None:
                    result[base_name] = df

        # Check for dictionary outputs
        for key, value in source.items():
            if isinstance(value, dict):
                df = self.get_dataframe(key)
                if df is not None:
                    result[key] = df

        return result

    def compute(self, *args: InputType) -> OutputType:
        """Compute the model outputs based on inputs passed as arguments."""
        self._compute(*args)

    def _compute(self, *args: InputType) -> OutputType:
        """Compute the model outputs based on inputs passed as arguments.

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
        """Retrieve the computed outputs.

        Returns:
            dict: A dictionary of output names and their computed values.

        """
        return self.outputs

    def _create_wapped_compute_bwd(
            self,
            output_name: str,
            input_names: Union[str, list] = None,
            all_inputs: bool = False,
    ) -> Callable | list[Callable]:
        """Create wrapped compute functions for a specific output and inputs.

        Args:
            output_name (str): The name of the output.
            input_names (str, list): The name of the input. Can be a list of inputs.
            all_inputs (bool): Whether to compute the derivative with respect to all
            inputs.

        Returns:
            (Callable, list[Callable]): A single wrapped compute function or a list of
            the wrapped compute functions for each input.

        """
        # Make sure either input_names or all_inputs is provided
        if input_names is None and all_inputs is False:
            msg = "Either input_names or all_inputs must be provided."
            raise ValueError(msg)

        # Ensure input_names is a list
        if isinstance(input_names, str):
            input_names = [input_names]

        wrapped_computes = None

        if all_inputs:

            def wrapped_compute(args: InputType) -> OutputType:
                temp_inputs = deepcopy(self.inputs)
                self.inputs = args
                self.compute()
                self.inputs = temp_inputs
                return self.outputs[output_name]

            wrapped_computes = wrapped_compute

        else:
            wrapped_computes = []

            for input_name in input_names:
                if isinstance(self.inputs[input_name], dict):

                    def wrapped_compute(
                            *args: InputType,
                            input_name: str = input_name,
                    ) -> OutputType:
                        temp_inputs = deepcopy(self.inputs)
                        for i, col in enumerate(self.inputs[input_name].keys()):
                            self.inputs[input_name][col] = args[i]
                        self.compute()
                        self.inputs = temp_inputs
                        return self.outputs[output_name]
                else:

                    def wrapped_compute(
                            arg: InputType,
                            input_name: str = input_name,
                    ) -> OutputType:
                        temp_inputs = deepcopy(self.inputs)
                        self.inputs[input_name] = arg
                        self.compute()
                        self.inputs = temp_inputs
                        return self.outputs[output_name]

                wrapped_computes.append(wrapped_compute)

        return wrapped_computes

    def compute_partial_bwd(
            self, output_name: str, input_names: str | list, all_inputs: bool = False
    ) -> (
            npt.NDArray[np.float64]
            | dict[str, npt.NDArray[np.float64] | dict[str, npt.NDArray[np.float64]]]
    ):
        """Compute the partial derivative of an output with respect to an input or all inputs.

        Args:
            output_name (str): The name of the output to compute the derivative for.
            input_names (Union[str, list]): The name or list of names of the input(s)
                                            with respect to which the derivative is computed.
            all_inputs (bool): Flag indicating whether to compute the derivative with respect
                               to all inputs at once.

        Returns:
            Union[npt.NDArray[np.float64], dict[str, Union[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]]]:
                The computed partial derivative(s) as a NumPy ndarray or a dictionary of arrays.

        """
        # pylint: disable=E1120

        is_single = False
        if isinstance(input_names, str):
            is_single = True
            input_names = [input_names]

        result = {}

        # Create wrapped compute functions making sure only asking for all inputs if using jax
        wrapped_computes = self._create_wapped_compute_bwd(
            output_name,
            input_names,
            all_inputs=all_inputs if self._ad_backend == "jax" else False,
        )

        # If all_inputs is True, compute the jacobian using all inputs at once
        if all_inputs:
            wrapped_compute = wrapped_computes
            jacobian_func = self.__jacobian(wrapped_compute)
            result = jacobian_func(self.inputs)

        else:  # If not, compute the jacobian for each input
            for wrapped_compute, input_name in zip(wrapped_computes, input_names):
                if isinstance(self.inputs[input_name], dict):  # For DataFrame inputs
                    jacobians = {}
                    argnum_kword = (
                        "argnum" if self._ad_backend == "autograd" else "argnums"
                    )
                    for col in self.inputs[input_name]:
                        jacobian_func = self.__jacobian(
                            wrapped_compute,
                            **{
                                argnum_kword: list(
                                    self.inputs[input_name].keys()
                                ).index(
                                    col,
                                ),
                            },
                        )
                        jacobians[col] = jacobian_func(
                            *self.inputs[input_name].values()
                        )

                    result[input_name] = jacobians

                else:  # For other inputs
                    jacobian_func = self.__jacobian(wrapped_compute)
                    result[input_name] = jacobian_func(self.inputs[input_name])

            if is_single:
                return result[input_names[0]]

        return result

    def compute_partial_all_inputs(self, output_name: str) -> dict:
        """Compute the Jacobian of the model output with respect to all inputs.

        Computes the Jacobian of the model output with respect to all inputs. This
        is useful for computing the Jacobian when the model has multiple inputs and
        you want to get the Jacobian with respect to all of them at once.

        Args:
            output_name (str):The name of the output of the model for which to compute
            the Jacobian.

        Returns:
            result (dict): A dictionary where the keys are the names of the inputs and
            the values are the Jacobians of the output with respect to the inputs.

        """
        if self._ad_backend == "autograd":
            result = {}
            for key in self.inputs:
                partial = self.compute_partial(output_name, key)
                result[key] = partial
        else:  # JAX
            result = self.compute_partial(
                output_name, list(self.inputs.keys()), all_inputs=True
            )

        return result

    def compute_partial_numeric(
            self,
            output_name: str,
            input_name: str,
            method: str = "complex_step",
            epsilon: float = 1e-8,
    ) -> dict[str, Union[np.ndarray, float, bool]]:
        """Compute the partial derivative of an output with respect to an input using a numerical method.

        Args:
            output_name (str): The name of the output to compute the derivative for.
            input_name (str): The name of the input with respect to which the derivative is computed.
            method (Literal["finite_differences", "complex_step"]): The numerical method to use.
        """

        def finite_difference(f, x, i, eps=epsilon):
            x_plus = x.copy()
            x_plus[i] += eps
            return (f(x_plus) - f(x)) / eps

        def complex_step(f, x, i, eps=epsilon):
            x_complex = x.copy().astype(complex)
            x_complex[i] += 1j * eps
            return np.imag(f(x_complex)) / eps

        # Prepare for numerical approximation
        wrapped_compute = self._create_wapped_compute_bwd(output_name, input_name)[0]

        if isinstance(self.inputs[input_name], dict):
            numerical = {}
            for col, value in self.inputs[input_name].items():
                if np.isscalar(value):

                    def f(x, col: str = col) -> Callable:
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

                        def f(x, col: str = col) -> Callable:
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

        return numerical

    def check_partial(
            self,
            output_name: str,
            input_name: str,
            method: str = "complex_step",
            epsilon: float = 1e-8,
            rtol: float = 1e-5,
            atol: float = 1e-8,
    ) -> dict[str, Union[np.ndarray, float, bool]]:
        """Compare the partial derivative computed by compute_partial with a numerical approximation
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

        # Get the analytical partial derivative
        analytical = self.compute_partial(output_name, input_name)

        # Get the numerical partial derivative
        numerical = self.compute_partial_numeric(
            output_name, input_name, method=method, epsilon=epsilon
        )

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


if __name__ == "__main__":
    from contextlib import contextmanager
    from typing import Any, Callable

    class MyModel(DifferentiableModel):
        def compute(self) -> None:
            x = self.inputs["x"]
            y = self.inputs["y"]

            y_a = y["a"] ** 3
            y_b = y["b"] ** 3

            result = self.np.sum(x ** 2)
            result = result + self.np.sum(y_a)
            result = result + self.np.sum(y_b ** 3)

            self.outputs["result"] = result

    def replace_namespace_instance(instance, new_namespace):
        instance.__dict__["np"] = new_namespace

    # Usage example
    model = MyModel(flatten_dfs=False, ad_backend="autograd")

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

    result = model.check_partial("result", "x", method="complex_step")
    print(f"Analytical: {result['analytical']}")
    print(f"Numerical: {result['numerical']}")
    print(f"Max absolute error: {result['max_absolute_error']}")
    print(f"Max relative error: {result['max_relative_error']}")
    print(f"Within tolerance: {result['within_tolerance']}")

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
    df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0], "feature2": [4.0, 5.0, 6.0]})
    model.set_inputs({"data": df})
    model.compute()

    with timer_context("SINGLE ALL"):
        for i in range(100):
            jacobian_y = model.compute_partial_all_inputs("other:value")
        print("Jacobian multiple:", jacobian_y)

    for o in model.outputs:
        jacobian_y = model.compute_partial_all_inputs(o)
        print(f"Jacobian ({o}):", jacobian_y)

    # Get a specific DataFrame
    result_df = model.get_dataframe(
        "result"
    )  # DataFrame with 'squared' and 'sum' columns
    other_df = model.get_dataframe("other")  # DataFrame with 'value' column

    # Get all DataFrames
    all_dfs = model.get_dataframes()  # Dictionary with 'result' and 'other' keys

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

    # Get a specific DataFrame
    result_df = model.get_dataframe(
        "result"
    )  # DataFrame with 'squared' and 'sum' columns
    single_value_df = model.get_dataframe("single_value")  # Returns None

    # Get all DataFrames
    all_dfs = model.get_dataframes()  # Dictionary with only 'result' key

    # %%
    class MyModel(DifferentiableModel):
        def compute(self) -> None:
            x = self.inputs["x"]
            y = self.inputs["y"]

            y_a = y["a"] ** 3
            y_b = y["b"] ** 3

            result = self.np.sum(x ** 2)
            result = result + self.np.sum(y_a)
            result = result + self.np.sum(y_b ** 3)

            self.outputs["result"] = result

    # %%
    # Usage example
    model = MyModel(flatten_dfs=False, ad_backend="autograd")

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

    result = model.check_partial("result", "x", method="complex_step")
    print(f"Analytical: {result['analytical']}")
    print(f"Numerical: {result['numerical']}")
    print(f"Max absolute error: {result['max_absolute_error']}")
    print(f"Max relative error: {result['max_relative_error']}")
    print(f"Within tolerance: {result['within_tolerance']}")

    result = model.check_partial("result", "y", method="complex_step")
    print(f"Analytical: {result['analytical']}")
    print(f"Numerical: {result['numerical']}")
    print(f"Max absolute error: {result['max_absolute_error']}")
    print(f"Max relative error: {result['max_relative_error']}")
    print(f"Within tolerance: {result['within_tolerance']}")

    # %%
    # Example with flatten_dfs=True
    class FlatModule(DifferentiableModel):
        def _compute(self):
            x = self.inputs["data:feature1"]
            y = self.inputs["data:feature2"]

            # Create flattened outputs
            self.outputs["result:squared"] = x ** 2
            self.outputs["result:sum"] = x + y
            self.outputs["other:value"] = x * y

    model = FlatModule(flatten_dfs=True)
    df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0], "feature2": [4.0, 5.0, 6.0]})
    model.set_inputs({"data": df})
    model.compute()

    with timer_context("SINGLE ALL"):
        for i in range(100):
            jacobian_y = model.compute_partial_all_inputs("other:value")
        print("Jacobian multiple:", jacobian_y)

    for o in model.outputs:
        jacobian_y = model.compute_partial_all_inputs(o)
        print(f"Jacobian ({o}):", jacobian_y)

    # Get a specific DataFrame
    result_df = model.get_dataframe("result")  # DataFrame with 'squared' and 'sum' columns
    other_df = model.get_dataframe("other")  # DataFrame with 'value' column

    # Get all DataFrames
    all_dfs = model.get_dataframes()  # Dictionary with 'result' and 'other' keys

    # %%
    # Example with flatten_dfs=False
    class DictModule(DifferentiableModel):
        def _compute(self):
            x = self.inputs["data"]["feature1"]
            y = self.inputs["data"]["feature2"]

            # Create dictionary outputs
            self.outputs["result:squared"] = x ** 2
            self.outputs["result:sum"] = x + y
            self.outputs["single_value"] = x.mean()  # This won't be converted to DataFrame

    model = DictModule(flatten_dfs=False)
    model.set_inputs({"data": df})
    model.compute()

    for o in model.outputs:
        j = model.compute_partial_all_inputs(o)
        print(f"Jacobian ({o}):", j)

    j = model.compute_partial("result:sum", "data")
    print("Jacobian (result:sum):", j)

    # Get a specific DataFrame
    result_df = model.get_dataframe("result")  # DataFrame with 'squared' and 'sum' columns
    single_value_df = model.get_dataframe("single_value")  # Returns None

    # Get all DataFrames
    all_dfs = model.get_dataframes()  # Dictionary with only 'result' key

    # %%
    class DictModule(DifferentiableModel):
        def _compute(self):
            # Extract inputs
            pollution_concentration = self.inputs["data:pollution_concentration"]
            emission_rate = self.inputs["data:emission_rate"]
            region_area = self.inputs["data:region_area"]

            # Validate inputs
            if (
                    pollution_concentration is None
                    or emission_rate is None
                    or region_area is None
            ):
                raise ValueError(
                    "All inputs (pollution_concentration, emission_rate, region_area) must be provided."
                )

            if self.np.any(region_area <= 0):
                raise ValueError("Region area must be positive for all elements.")

            # Step 1: Calculate pollution density
            pollution_density = self.calculate_pollution_density(
                pollution_concentration, region_area
            )

            # Step 2: Calculate radiative forcing for each density
            radiative_forcing = self.calculate_radiative_forcing(pollution_density)

            # Step 3: Compute temperature change based on thresholds
            temperature_change = self.calculate_temperature_change(radiative_forcing)

            # Step 4: Adjust for emission rate
            adjusted_temperature_change = self.adjust_for_emission_rate(
                temperature_change, emission_rate
            )

            # Store results in the outputs dictionary
            self.outputs["pollution_density"] = pollution_density
            self.outputs["radiative_forcing"] = radiative_forcing
            self.outputs["temperature_change"] = adjusted_temperature_change

        def calculate_pollution_density(self, concentration, area):
            """Calculates pollution density per unit area."""
            return concentration / area

        def calculate_radiative_forcing(self, pollution_density):
            """Calculates radiative forcing based on pollution density."""
            return self.np.where(
                pollution_density < 10,
                5.35 * self.np.log1p(pollution_density),
                6.0 * self.np.log1p(pollution_density),
            )

        def calculate_temperature_change(self, radiative_forcing):
            """Calculates temperature change using climate sensitivity and a threshold."""
            climate_sensitivity = 0.8  # K per W/m²
            temperature_change = radiative_forcing * climate_sensitivity

            # Apply temperature caps for extreme forcing
            temperature_cap = 5.0  # Max temperature increase in K
            return self.np.minimum(temperature_change, temperature_cap)

        def adjust_for_emission_rate(self, temperature_change, emission_rate):
            """Adjusts temperature change based on emission rate."""
            return self.np.where(
                emission_rate > 0.5, temperature_change * 1.2, temperature_change * 0.9
            )

    # %%
    inputs = {}

    # Provide input values
    inputs["pollution_concentration"] = np.array([10.0, 20.0, 50.0])  # μg/m³
    inputs["emission_rate"] = np.array([1.0, 6.0, 3.0])  # tons per year
    inputs["region_area"] = np.array([100.0, 200.0, 300.0])  # km²

    data_df = pd.DataFrame(inputs)

    # %%
    model = DictModule(flatten_dfs=True, ad_backend="autograd")
    model.set_inputs({"data": data_df})
    model.compute()

    # %%
    model.outputs

    ic = print

    # %%
    for o in model.outputs:
        j = model.compute_partial_all_inputs(o)
        print(f"Jacobian ({o}):")
        ic(j)
        print("")

    # %%
    for o in model.outputs:
        j = model.compute_partial(o, [f"data:{i}" for i in inputs])
        print(f"Jacobian ({o}):")
        ic(j)
        print("")

    # %%
    for o in model.outputs:
        with timer_context(f"CHECK PARTIAL {o}"):
            result = model.check_partial(
                o, "data:pollution_concentration", method="complex_step"
            )
            ic(f"Analytical: {result['analytical']}")
            ic(f"Numerical: {result['numerical']}")
            print(f"Max absolute error: {result['max_absolute_error']}")
            print(f"Max relative error: {result['max_relative_error']}")
            print(f"Within tolerance: {result['within_tolerance']}")
