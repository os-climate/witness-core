# **DifferentiableModel Tutorial**

This tutorial explains how to use the `DifferentiableModel` class, which provides a framework for creating differentiable models in Python. We'll cover the basics, best practices, and provide examples of how to implement and use this model, including different ways of handling DataFrames.


## Table of Contents

- [**DifferentiableModel Tutorial**](#differentiablemodel-tutorial)
  - [Table of Contents](#table-of-contents)
  - [Basic Usage](#basic-usage)
  - [Handling DataFrames](#handling-dataframes)
    - [Flattened DataFrames](#flattened-dataframes)
    - [Dictionary-style DataFrames](#dictionary-style-dataframes)
  - [Computing Partial Derivatives](#computing-partial-derivatives)
  - [Checking Partial Derivatives](#checking-partial-derivatives)
  - [Do's and Don'ts](#dos-and-donts)
    - [Do's](#dos)
    - [Don'ts](#donts)
  - [Examples](#examples)
    - [Flattened DataFrame Example](#flattened-dataframe-example)
    - [Dictionary-style DataFrame Example](#dictionary-style-dataframe-example)
    - [Single File Example](#single-file-example)
    - [Multi-File Example](#multi-file-example)


## Basic Usage

To use the `DifferentiableModel`, follow these steps:

1. Subclass `DifferentiableModel`
2. Implement the `compute` method
3. Set inputs
4. Compute the model
5. Retrieve outputs
6. Compute partial derivatives (optional)

Here's a simple example:

```python
import autograd.numpy as np
from differentiable_model import DifferentiableModel

class MyModel(DifferentiableModel):
    def compute(self) -> None:
        x = self.inputs["x"]
        y = self.inputs["y"]
        result = np.sum(x**2) + np.sum(y**2)
        self.outputs["result"] = np.array([result])

# Create an instance
model = MyModel()

# Set inputs
inputs = {
    "x": np.array([1.0, 2.0, 3.0]),
    "y": np.array([4.0, 5.0, 6.0])
}
model.set_inputs(inputs)

# Compute the model
model.compute()

# Get outputs
outputs = model.get_outputs()
print("Output:", outputs["result"])

# Compute partial derivative
jacobian_x = model.compute_partial("result", "x")
print("Jacobian_x:", jacobian_x)
```

## Handling DataFrames

The `DifferentiableModel` class provides two ways to handle DataFrame inputs: flattened and dictionary-style.

### Flattened DataFrames

When `flatten_dfs=True` (default), DataFrames are flattened into separate arrays with keys as 'dataframe_name:column_name'.

```python
class FlatModule(DifferentiableModel):
    def compute(self):
        x = self.inputs["data:feature1"]
        y = self.inputs["data:feature2"]
        self.outputs["result:squared"] = x**2
        self.outputs["result:sum"] = x + y

model = FlatModule(flatten_dfs=True)
df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
model.set_inputs({"data": df})
model.compute()
```

### Dictionary-style DataFrames

When `flatten_dfs=False`, DataFrames are converted to dictionaries of arrays.

```python
class DictModule(DifferentiableModel):
    def compute(self):
        x = self.inputs["data"]["feature1"]
        y = self.inputs["data"]["feature2"]
        self.outputs["result"] = {"squared": x**2, "sum": x + y}

model = DictModule(flatten_dfs=False)
model.set_inputs({"data": df})
model.compute()
```

## Computing Partial Derivatives

The `DifferentiableModel` class provides several methods for computing partial derivatives:

1. `compute_partial(output_name, input_name)`: Computes the partial derivative of a specific output with respect to a specific input.
2. `compute_partial_all_inputs(output_name)`: Computes the partial derivatives of a specific output with respect to all inputs.
3. `compute_partial_multiple(output_name, input_names)`: Computes the partial derivatives of a specific output with respect to multiple inputs efficiently.

```python
jacobian_x = model.compute_partial("result", "x")
jacobian_all = model.compute_partial_all_inputs("result")
jacobian_multiple = model.compute_partial_multiple("result", ["x", "y"])
```

## Checking Partial Derivatives

The `check_partial` method allows you to compare the analytical partial derivative with a numerical approximation:

```python
result = model.check_partial("result", "x", method="complex_step")
print(f"Analytical: {result['analytical']}")
print(f"Numerical: {result['numerical']}")
print(f"Max absolute error: {result['max_absolute_error']}")
print(f"Max relative error: {result['max_relative_error']}")
print(f"Within tolerance: {result['within_tolerance']}")
```


## Do's and Don'ts

Here is a non-exaustive list of do's and don'ts when creating an automatically differentiable model. (more info can be found in HIPS autograd tutorial here: https://github.com/HIPS/autograd/blob/master/docs/tutorial.md)

### Do's

1. **Implement `compute` method**: Always implement the `compute` method in your subclass.
2. **Use numpy arrays**: Use numpy arrays for numerical computations to ensure compatibility with autograd.
3. **Handle DataFrame inputs carefully**: When using pandas DataFrames, make sure to convert them to numpy arrays in the `compute` method.
4. **Use `self.inputs` and `self.outputs`**: Access input values via `self.inputs` and store results in `self.outputs`.
5. **Validate inputs**: Check that inputs are of the expected type and shape before computations.

### Don'ts

1. **Don't modify inputs directly**: Avoid modifying `self.inputs` within the `compute` method.
2. **Don't use non-differentiable operations**: Avoid operations that are not differentiable (e.g., integer operations) if you plan to compute gradients.
3. **Don't return values from `compute`**: Store results in `self.outputs` instead of returning them.
4. **Don't use external state**: Avoid relying on external state that isn't passed through `self.inputs` or `self.parameters`.
5. **Don't use in-place operations**: Avoid in-place modifications of arrays, as they can interfere with autograd's ability to compute gradients.

## Examples

### Flattened DataFrame Example

```python
class FlatModule(DifferentiableModel):
    def compute(self):
        x = self.inputs["data:feature1"]
        y = self.inputs["data:feature2"]
        self.outputs["result:squared"] = x**2
        self.outputs["result:sum"] = x + y
        self.outputs["other:value"] = x * y

model = FlatModule(flatten_dfs=True)
df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
model.set_inputs({"data": df})
model.compute()

jacobian_multiple = model.compute_partial_multiple("other:value", ["data:feature1", "data:feature2"])
print("Jacobian multiple:", jacobian_multiple)

jacobian_all = model.compute_partial_all_inputs("other:value")
print("Jacobian all:", jacobian_all)

# Get a specific DataFrame
result_df = model.get_dataframe('result')  # DataFrame with 'squared' and 'sum' columns
other_df = model.get_dataframe('other')   # DataFrame with 'value' column

# Get all DataFrames
all_dfs = model.get_dataframes()  # Dictionary with 'result' and 'other' keys
```

### Dictionary-style DataFrame Example

```python
class DictModule(DifferentiableModel):
    def compute(self):
        x = self.inputs["data"]["feature1"]
        y = self.inputs["data"]["feature2"]
        self.outputs["result"] = {"squared": x**2, "sum": x + y}
        self.outputs["single_value"] = x.mean()  # This won't be converted to DataFrame

model = DictModule(flatten_dfs=False)
model.set_inputs({"data": df})
model.compute()

for output_name in model.outputs:
    jacobian = model.compute_partial_all_inputs(output_name)
    print(f"Jacobian ({output_name}):", jacobian)

jacobian_sum = model.compute_partial("result:sum", "data")
print("Jacobian (result:sum):", jacobian_sum)

# Get a specific DataFrame
result_df = model.get_dataframe('result')  # DataFrame with 'squared' and 'sum' columns
single_value_df = model.get_dataframe('single_value')  # Returns None

# Get all DataFrames
all_dfs = model.get_dataframes()  # Dictionary with only 'result' key
```

### Single File Example

Here's an example where everything is in one file:

```python
import autograd.numpy as np
import pandas as pd
from differentiable_model import DifferentiableModel

class ComplexModel(DifferentiableModel):
    def compute(self) -> None:
        x = self.inputs["x"]
        y = self.inputs["y"]
        z = self.inputs["z"]
        
        # Perform complex computations
        intermediate1 = np.sum(x**2) + np.sum(y["a"]**3)
        intermediate2 = np.prod(z) * np.mean(y["b"])
        
        result = intermediate1 * intermediate2
        self.outputs["result"] = np.array([result])

# Usage
model = ComplexModel()

inputs = {
    "x": np.array([1.0, 2.0, 3.0]),
    "y": pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}),
    "z": np.array([0.1, 0.2, 0.3])
}
model.set_inputs(inputs)

model.compute()
outputs = model.get_outputs()
print("Output:", outputs["result"])

# Compute partial derivatives
jacobian_x = model.compute_partial("result", "x")
print("Jacobian_x:", jacobian_x)

jacobian_all = model.compute_partial_all_inputs("result")
print("Jacobian_all:", jacobian_all)
```

### Multi-File Example

For larger projects, you might want to split your code into multiple files. Here's an example:

File: `computations.py`
```python
import autograd.numpy as np

def compute_intermediate1(x, y_a):
    return np.sum(x**2) + np.sum(y_a**3)

def compute_intermediate2(z, y_b):
    return np.prod(z) * np.mean(y_b)
```

File: `complex_model.py`
```python
import autograd.numpy as np
from differentiable_model import DifferentiableModel
from computations import compute_intermediate1, compute_intermediate2

class ComplexModel(DifferentiableModel):
    def compute(self) -> None:
        x = self.inputs["x"]
        y = self.inputs["y"]
        z = self.inputs["z"]
        
        intermediate1 = compute_intermediate1(x, y["a"])
        intermediate2 = compute_intermediate2(z, y["b"])
        
        result = intermediate1 * intermediate2
        self.outputs["result"] = np.array([result])
```

File: `main.py`
```python
import autograd.numpy as np
import pandas as pd
from complex_model import ComplexModel

# Usage
model = ComplexModel()

inputs = {
    "x": np.array([1.0, 2.0, 3.0]),
    "y": pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}),
    "z": np.array([0.1, 0.2, 0.3])
}
model.set_inputs(inputs)

model.compute()
outputs = model.get_outputs()
print("Output:", outputs["result"])

# Compute partial derivatives
jacobian_x = model.compute_partial("result", "x")
print("Jacobian_x:", jacobian_x)

jacobian_all = model.compute_partial_all_inputs("result")
print("Jacobian_all:", jacobian_all)
```

This multi-file structure allows for better organization of code, especially for more complex models or when reusing computational functions across different models.
