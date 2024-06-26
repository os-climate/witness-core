# How to Contribute?

First off, thanks for considering contributing!

All types of contributions are encouraged and valued.

## Summary with Navigation Links

- [1. Quick Overview](#1-quick-overview)
- [2. Github workflow](#2-github-workflow)
- [3. How Should I Develop My Ideas?](#3-how-should-i-develop-my-ideas)
  - [3.0. What is a coupling variable?](#30-what-is-a-coupling-variable)
  - [3.1. Creating a Model & a New Discipline](#31-creating-a-model--a-new-discipline)
  - [3.2. Existing Discipline Modifications](#32-existing-discipline-modifications)
    - [3.2.1. Adding a Non-coupling variable or a New Post-processing Graph](#321-adding-a-non-coupling-variable-or-a-new-post-processing-graph)
    - [3.2.2. Adding a Coupling variable](#322-adding-a-coupling-variable)
    - [3.2.3. I Want to Modify How an Output Is Computed](#323-i-want-to-modify-how-an-output-is-computed)
  - [3.3. New Process and/or New Usecase](#33-new-process-andor-new-usecase)
  - [3.4. Existing Process Modifications](#34-existing-process-modifications)

## 1. Quick Overview

The integration of a contribution is done in two steps:

1. **Contributor side:** the contributor pushes its developments to a dedicated place.
2. **Integrator side:** two validations are necessary:
   - **Code Validity:** the developments haven't caused any regression of the code (assessed by a _Witness integrator_).
   - **Scientific Content Validity:** the nature of the contribution makes sense on a scientific level (assessed by _Witness model reviewer_).

> **Important:** If for your development, you use data found online, make sure it is open source, and when you use it, cite your sources and where you found it.

## 2. Github workflow

- Create your branch(es) from the `develop` branch and initiate development.
- Regularly merge `develop` into your branch(es).
- Ensure that your local tests match the test state on the `develop` branch for each repository intended for pushing. The tests to pass include:
  - `l0` (including `headers_test`)
  - `l1`
  - Usecases (or `l1s_test_all_usecases`)
  - Pylint
- Create a pull request of your work to the `develop` branch(es) of the repo(s) you wish to push.

## 3. How Should I Develop My Ideas?

Depending on the nature of your contribution, here is the way of working to implement your ideas:

### 3.0. What is a coupling variable?

_This notion will be important for the following._

> A **_coupling variable_** is a variable that is **an output** of a discipline **and an input** for another, in a process.

### 3.1. Creating a Model & a New Discipline

> - [Check how to wrap a model here](https://sostrades-core.readthedocs.io/en/latest/how-to/wrap-model.html)
> - [Check how to test a discipline here](https://sostrades-core.readthedocs.io/en/latest/how-to/test-wrap.html)

**Documentation:** Create a markdown file `<my_model_name>.md`, stored in a `documentation/` folder next to the discipline, explaining how your model works.

**Tests:**
In this case, you should create a new `l0_test_<my_model_name>.py` file to test your model/discipline.
In the `l0` test file, the test class should test at least:

- a run of the discipline
- a retrieval of the post-processings

**I also coded gradients!**
If you also coded gradients in the discipline, create a new `l1_test_gradient_<my_model_name>.py` to test the gradients of your discipline.
You should be testing the gradients of all coupling outputs with respect to all coupling inputs.

### 3.2. Existing Discipline Modifications

#### 3.2.1. Adding a Non-coupling variable or a New Post-processing Graph

> - [Check how to add post-processings here](https://sostrades-core.readthedocs.io/en/latest/how-to/create-postprocessing.html)

No new test creation required. Still, all tests (`l0`, `l1`, and use cases) should be OK after modifications.

In the case of adding a new input, some tests might require fixes: specify values for new missing inputs.

#### 3.2.2. Adding a Coupling variable

**Documentation:** Update the markdown documentation file stored in a `documentation/` folder next to the discipline.

**Tests:**

- Compute its gradients in the method `compute_sos_jacobian()` of the discipline.
- Complete its gradient test (`l1_test_gradient_<model_name>.py`) by also checking for this input/output.

#### 3.2.3. I Want to Modify How an Output Is Computed

If you want to modify the way a certain value is computed, it is better to let the possibility to use the old formula instead of losing it.
For that, you can add a flag (a boolean) or a list of options to select how to compute the desired value in the model.
Also, after adding this flag/option, make sure the default parameters are set to compute the value as it was before contributing.
It is only later, during discussions with model reviewers, that the default settings might be changed.

If the output you modify is a coupling variable, also compute its gradients in accordance with the selected option.

**Documentation:** Update the markdown documentation file stored in a `documentation/` folder next to the discipline.

**Tests:**

- Find the existing `l0` test of the discipline.
- Create a new test function in the test class and set values to use your new method.
- If the modified output is a coupling variable in Witness:
  - Find the existing `l1` test of the discipline.
  - Create a new test function in the test class and set values to use your new method in the model.
  - Make sure the gradients are OK with the new selected method.

### 3.3. New Process and/or New Usecase

> - [Check how to create a process here](https://sostrades-core.readthedocs.io/en/latest/how-to/create-process.html)
> - [Check how to create a usecase here](https://sostrades-core.readthedocs.io/en/latest/how-to/create-usecase.html)

You created a new process? Create a `usecase_<my_comment>.py` file that goes in the same folder as the `process.py` file.
Your use case will rely on the process. You can test the use case by creating an instance of your `Study` class and call `my_study.test()` method.

### 3.4. Existing Process Modifications

This needs further discussions with a Witness model reviewer.
