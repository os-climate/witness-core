# **DifferentiableModel Tutorial**

This tutorial explains how to use the `DifferentiableModel` class, which provides a framework for creating differentiable models in Python. We'll cover the basics, best practices, and provide examples of how to implement and use this model.

## Table of Contents

- [**DifferentiableModel Tutorial**](#differentiablemodel-tutorial)
  - [Table of Contents](#table-of-contents)
  - [Basic Usage](#basic-usage)
  - [Do's and Don'ts](#dos-and-donts)
    - [Do's](#dos)
    - [Don'ts](#donts)
  - [Examples](#examples)
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
import numpy as np
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
