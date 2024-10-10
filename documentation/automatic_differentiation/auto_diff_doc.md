# **Automatic Differentiation with `autograd`**

## **Table of Contents**
- [**Automatic Differentiation with `autograd`**](#automatic-differentiation-with-autograd)
  - [**Table of Contents**](#table-of-contents)
- [**Automatic Differentiation with `autograd`**](#automatic-differentiation-with-autograd-1)
  - [**Introduction**](#introduction)
    - [**What is automatic differentiation**](#what-is-automatic-differentiation)
      - [Resources on Automatic Differentiation (AD)](#resources-on-automatic-differentiation-ad)
    - [**Autograd**](#autograd)
  - [**Installation**](#installation)
  - [**Examples**](#examples)
    - [**Differentiating Functions of Single Float Arguments**](#differentiating-functions-of-single-float-arguments)
    - [**Differentiating Functions of Multiple Arguments (Floats)**](#differentiating-functions-of-multiple-arguments-floats)
    - [**Differentiating Functions with Numpy Arrays as Input**](#differentiating-functions-with-numpy-arrays-as-input)
    - [**Differentiating Functions with Mixed Types (Floats and Arrays)**](#differentiating-functions-with-mixed-types-floats-and-arrays)
    - [**Differentiating Functions Returning Numpy Arrays**](#differentiating-functions-returning-numpy-arrays)
    - [**Differentiating Functions with Multiple Inputs and Numpy Array Outputs**](#differentiating-functions-with-multiple-inputs-and-numpy-array-outputs)
    - [**Example: Function with Multiple Inputs and Numpy Array Output**](#example-function-with-multiple-inputs-and-numpy-array-output)
    - [**Handling Functions Returning Dictionaries**](#handling-functions-returning-dictionaries)
  - [**Understanding `grad`, `jacobian`, and `elementwise_grad`**](#understanding-grad-jacobian-and-elementwise_grad)
    - [**`grad`:**](#grad)
    - [**`jacobian`:**](#jacobian)
    - [**`elementwise_grad`:**](#elementwise_grad)
  - [**Practical Examples for `elementwise_grad`**](#practical-examples-for-elementwise_grad)
    - [**Example 1: Vectorized Activation Function in Neural Networks**](#example-1-vectorized-activation-function-in-neural-networks)
    - [**Example 2: Custom Element-wise Functions in Signal Processing**](#example-2-custom-element-wise-functions-in-signal-processing)
    - [**Example 3: Pixel-wise Gradient in Image Processing**](#example-3-pixel-wise-gradient-in-image-processing)
    - [**Example 4: Gradient of Element-wise Loss Function in Machine Learning**](#example-4-gradient-of-element-wise-loss-function-in-machine-learning)

---

# **Automatic Differentiation with `autograd`**

## **Introduction**

### **What is automatic differentiation**

Automatic differentiation (AD) is a set of techniques to numerically evaluate the derivative of a function specified by a computer program. It is different from both symbolic differentiation and numerical differentiation (finite differences).

AD works by breaking down complex functions into a sequence of elementary operations (like addition, multiplication, sin, cos, etc.) and applying the chain rule systematically to these operations. This allows AD to compute derivatives accurately to machine precision.

There are two main modes of AD:

1. **Forward Mode**: 
   - Computes the derivative alongside the function evaluation.
   - Efficient for functions with few inputs and many outputs.
   - Equation: If y = f(x), then dy/dx is computed along with y.

2. **Reverse Mode**: 
   - Computes the derivative backwards from the output to the inputs.
   - Efficient for functions with many inputs and few outputs.
   - Commonly used in machine learning (backpropagation).
   - Equation: If y = f(x), first y is computed, then dy/dx is computed backwards.

Example of forward mode for y = x^2:
1. v₁ = x
2. v₂ = v₁ * v₁
3. y = v₂

Corresponding derivative computations:
1. dv₁/dx = 1
2. dv₂/dx = dv₁/dx * v₁ + v₁ * dv₁/dx = 1 * x + x * 1 = 2x
3. dy/dx = dv₂/dx = 2x

Reverse mode would compute these derivatives in reverse order.

AD provides exact derivatives (up to floating-point precision) and is more computationally efficient than numerical methods, especially for functions with many variables. 

#### Resources on Automatic Differentiation (AD)

Compiled by Peter Sharpe (https://github.com/peterdsharpe/AeroSandbox/blob/master/tutorial/10%20-%20Miscellaneous/03%20-%20Resources%20on%20Automatic%20Differentiation.md)

-----

Lectures, Videos, GitHub repos, blog posts:

* [YouTube: "What is Automatic Differentiation"](https://www.youtube.com/watch?v=wG_nF1awSSY) - a "3Blue1Brown"-style video introducing AD
* [Medium: "Automatic Differentiation Step by Step"](https://marksaroufim.medium.com/automatic-differentiation-step-by-step-24240f97a6e6)
* [GitHub: "Differentiation for Hackers"](https://github.com/MikeInnes/diff-zoo) - includes runnable examples in Julia
* [Lecture: "Automatic differentiation"](http://videolectures.net/deeplearning2017_johnson_automatic_differentiation/) - by Matthew J. Johnson, a huge contributor in the current AD landscape (developer of autograd, JAX).
* [Lecture: "Intuition behind reverse mode algorithmic differentiation"](https://youtu.be/twTIGuVhKbQ) - by Joris Gillis, one of the developers of CasADi (the AD library underneath AeroSandbox).
* [Blog: "Reverse-mode automatic differentiation: a tutorial"](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation)
* [Lecture: "Large-Scale Multidisciplinary Design Optimization of Aerospace Systems"](https://www.pathlms.com/siam/courses/479/sections/678/thumbnail_video_presentations/5169) - by Joaquim Martins. Section on AD starts at ~17:00.

Academic literature:

* Griewank and Walther, "Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation", 2008. Great comprehensive overview.
* ["Automatic Differentiation in Machine Learning: a Survey" on ArXiV by Baydin et. al.](https://arxiv.org/abs/1502.05767) is another great read.
* ["Automatic differentiation of algorithms"](https://www.sciencedirect.com/science/article/pii/S0377042700004222?via%3Dihub) by Bartholomew-Biggs, et. al.

### **Autograd**

`autograd` is a Python package for automatic differentiation, making it easy to compute derivatives of functions using Python and NumPy. This tutorial will guide you through the basics of using `autograd` to differentiate functions of single and multiple variables, both floats and numpy arrays, and even functions returning more complex structures like dictionaries and numpy arrays.

We'll cover:
1. Differentiating functions with single float arguments
2. Differentiating functions with multiple arguments (both floats and numpy arrays)
3. Handling functions with mixed input types and multiple outputs (e.g., numpy arrays or dictionaries)
4. Understanding the difference between `grad`, `jacobian`, and `elementwise_grad`
5. Practical examples for using `elementwise_grad`

---

## **Installation**

To install `autograd`, use pip:

```bash
pip install autograd
``` 

---
## **Examples**

### **Differentiating Functions of Single Float Arguments**

To start, let's differentiate a simple function \( f(x) = x^2 \).

```python
import autograd.numpy as np
from autograd import grad

# Define the function
def f(x):
    return x**2

# Create the gradient function
grad_f = grad(f)

# Test the gradient function
x = 3.0
print(f"The gradient of f at x = {x} is {grad_f(x)}")
```

---

### **Differentiating Functions of Multiple Arguments (Floats)**

Now let's handle a function with multiple float inputs, like \( g(x, y) = x^2 + y^2 \).

```python
# Define the function
def g(x, y):
    return x**2 + y**2

# Create the gradient function with respect to x
grad_g_x = grad(g, 0)

# Create the gradient function with respect to y
grad_g_y = grad(g, 1)

# Test the gradient function
x, y = 3.0, 4.0
print(f"The gradient of g with respect to x at (x, y) = ({x}, {y}) is {grad_g_x(x, y)}")
print(f"The gradient of g with respect to y at (x, y) = ({x}, {y}) is {grad_g_y(x, y)}")
```

---

### **Differentiating Functions with Numpy Arrays as Input**

Let's extend to functions that accept `numpy` arrays as input and return a scalar.

```python
# Define a function that operates on numpy arrays
def h(arr):
    return np.sum(arr**2)

# Create the gradient function
grad_h = grad(h)

# Test the gradient function
arr = np.array([1.0, 2.0, 3.0])
print(f"The gradient of h at arr = {arr} is {grad_h(arr)}")
```

---

### **Differentiating Functions with Mixed Types (Floats and Arrays)**

Consider a function that accepts both a float and a numpy array:

```python
# Define a function that accepts a float and a numpy array
def k(x, arr):
    return x * np.sum(arr**2)

# Create gradient function with respect to x
grad_k_x = grad(k, 0)

# Create gradient function with respect to arr
grad_k_arr = grad(k, 1)

# Test the gradient function
x = 2.0
arr = np.array([1.0, 2.0, 3.0])
print(f"The gradient of k with respect to x at (x, arr) = ({x}, {arr}) is {grad_k_x(x, arr)}")
print(f"The gradient of k with respect to arr at (x, arr) = ({x}, {arr}) is {grad_k_arr(x, arr)}")
```

---

### **Differentiating Functions Returning Numpy Arrays**

Sometimes, a function may return a `numpy` array. To compute the Jacobian (matrix of partial derivatives), we use `autograd.jacobian` instead of `grad`.

Here's an example where a function returns a numpy array:

```python
from autograd import jacobian

# Define a function that returns a numpy array
def vector_func(arr):
    return arr**2

# Create the Jacobian function
jacobian_vector_func = jacobian(vector_func)

# Test the Jacobian function
arr = np.array([1.0, 2.0, 3.0])
print(f"The Jacobian of vector_func at arr = {arr} is:\n{jacobian_vector_func(arr)}")
```

---

### **Differentiating Functions with Multiple Inputs and Numpy Array Outputs**

Now, let's handle the case where a function has **multiple inputs** and returns a **numpy array**. This scenario is common when working with functions that return vector outputs, such as when implementing neural networks or vector-valued functions in optimization.

### **Example: Function with Multiple Inputs and Numpy Array Output**

Here’s an example function \( f(x, y) = [x + y, x \cdot y] \) that returns a numpy array:

```python
# Define a function that returns a numpy array and takes two inputs
def multi_input_array_output(x, y):
    return np.array([x + y, x * y])

# Create the Jacobian function with respect to the first argument (x)
jacobian_x = jacobian(multi_input_array_output, 0)

# Create the Jacobian function with respect to the second argument (y)
jacobian_y = jacobian(multi_input_array_output, 1)

# Test the Jacobian functions
x, y = 2.0, 3.0
print(f"The Jacobian of multi_input_array_output with respect to x at (x, y) = ({x}, {y}) is:\n{jacobian_x(x, y)}")
print(f"The Jacobian of multi_input_array_output with respect to y at (x, y) = ({x}, {y}) is:\n{jacobian_y(x, y)}")
```

---

### **Handling Functions Returning Dictionaries**

Finally, let’s look at differentiating functions that return more complex structures, like dictionaries.

```python
# Define a function that returns a dictionary
def m(x):
    return {'result1': x**2, 'result2': x**3}

# Create the gradient function for 'result1'
grad_m_result1 = grad(lambda x: m(x)['result1'])

# Test the gradient function
x = 2.0
print(f"The gradient of m['result1'] at x = {x} is {grad_m_result1(x)}")
```

---

## **Understanding `grad`, `jacobian`, and `elementwise_grad`**

### **`grad`:**

- **Purpose**: Computes the gradient of a scalar-valued function with respect to its input(s). If a function returns a scalar and takes either a scalar or vector input, `grad` will return the gradient (derivative) with respect to the input(s).
  
- **Example**:
    ```python
    def f(x):
        return x**2
    
    grad_f = grad(f)
    ```

### **`jacobian`:**

- **Purpose**: Computes the **Jacobian matrix**, which is a matrix of all partial derivatives of a vector-valued function. The Jacobian gives a derivative of each output with respect to each input.
  
- **When to use**: Use `jacobian` when your function returns a vector (e.g., a numpy array) and you need the derivative of each element of the output with respect to each input variable.


  
- **Example**:
    ```python
    def vector_func(arr):
        return arr**2
    
    jacobian_vector_func = jacobian(vector_func)
    ```

### **`elementwise_grad`:**

- **Purpose**: Computes the gradient element-wise for each entry in an array independently. This is useful when you need the derivative of each element in a vector/matrix with respect to its corresponding input.
  
- **When to use**: Use `elementwise_grad` when you want to apply the gradient computation element-by-element in vectorized or matrix inputs, such as when differentiating element-wise activation functions, loss functions, or custom transformations.

---

## **Practical Examples for `elementwise_grad`**

### **Example 1: Vectorized Activation Function in Neural Networks**

```python
from autograd import elementwise_grad
import autograd.numpy as np

# Define the ReLU function (element-wise)
def relu(x):
    return np.maximum(0, x)

# Compute the element-wise gradient of the ReLU function
grad_relu = elementwise_grad(relu)

# Test the gradient function
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"ReLU(x): {relu(x)}")
print(f"Element-wise gradient of ReLU(x): {grad_relu(x)}")
```

---

### **Example 2: Custom Element-wise Functions in Signal Processing**

```python
# Define a custom element-wise function (e.g., tanh-like function)
def custom_func(x):
    return np.tanh(x) + 0.1 * np.sin(x)

# Compute the element-wise gradient of the custom function
grad_custom_func = elementwise_grad(custom_func)

# Test the gradient function on a signal (numpy array)
x = np.linspace(-2.0, 2.0, 5)
print(f"Custom function applied to x: {custom_func(x)}")
print(f"Element-wise gradient of the custom function: {grad_custom_func(x)}")
```

---

### **Example 3: Pixel-wise Gradient in Image Processing**

```python
# Define a pixel-wise transformation (e.g., sigmoid-like function)
def pixel_transform(x):
    return 1 / (1 + np.exp(-x))  # Sigmoid function

# Compute the element-wise gradient of the pixel transformation
grad_pixel_transform = elementwise_grad(pixel_transform)

# Simulate an image as a 2D numpy array
image = np.array([[0.1, 0.2, 0.3],
                  [0.4, 0.5, 0.6],
                  [0.7, 0.8, 0.9]])

print(f"Transformed image:\n{pixel_transform(image)}")
print(f"Element-wise gradient of the pixel transformation:\n{grad_pixel_transform(image)}")
```

---

### **Example 4: Gradient of Element-wise Loss Function in Machine Learning**

```python
# Define an element-wise loss function (e.g., squared error)
def loss(pred, target):
    return (pred - target) ** 2

# Compute the element-wise gradient of the loss function
grad_loss = elementwise_grad(loss)

# Test with prediction and target arrays
pred = np.array([2.5, 0.0, 2.1])
target = np.array([3.0, -0.5, 2.0])

print(f"Loss: {loss(pred, target)}")
print(f"Element-wise gradient of the loss: {grad_loss(pred, target)}")
```

---

This concludes the tutorial on using `autograd` for automatic differentiation. You've seen how to handle scalar, vector, and dictionary-based functions, as well as how to use `grad`, `jacobian`, and `elementwise_grad` effectively.

