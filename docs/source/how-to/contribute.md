# How to Contribute?

First off, thanks for considering contributing! 

All types of contributions are encouraged and valued.

### Quick Overview
The integration of a contribution is done in two steps:
1. **Contributor side:** the contributor pushes its developments to a dedicated place.
2. **Integrator side:** two validations are necessary:
    - **Code Validity:** the developments haven't caused any regression of the code (assessed by a *Witness integrator*).
    - **Scientific Content Validity:** the nature of the contribution makes sense on a scientific level (assessed by *Witness model reviewer*).

**Important:** If for your development, you use data found online, make sure it is open source, and when you use it, cite your sources and where you found it.

## Where and How Should I Develop?
- Create your branch(es) from the `develop` branch and initiate development.
- Regularly merge `develop` into your branch(es).
- Ensure that your local tests match the test state on the `develop` branch for each repository intended for pushing. The tests to pass include:
  - `l0` (including `headers_test`)
  - `l1`
  - Usecases (or `l1s_test_all_usecases`)
  - Pylint
- Create a pull request of your work to the `develop` branch(es) of the repo(s) you wish to push.

## How Should I Develop My Ideas?
Depending on the nature of your contribution, here is the way of working to implement your ideas:

### Creating a Model & a New Discipline
In this case, you should create a new `l0_test_<my_model_name>.py` file to test your model/discipline.
In the `l0` test file, the test class should test at least:
- a run of the discipline
- a retrieval of the post-processings

**I also coded gradients!**:
If you also coded gradients in the discipline, create a new `l1_test_gradient_<my_model_name>.py` to test the gradients of your discipline.
You should be testing the gradients of all coupling outputs with respect to all coupling inputs.

*Note:* a *coupling variable* is a variable that is an output of discipline and an input for another, in a process.

### Existing Discipline Modifications

#### Adding a Non-coupling Input/Output or a New Graph
No test creation/modification required. Still, all tests (`l0`, `l1`, and use cases) should be OK after modifications.

#### Adding a Coupling Input/Output
- Compute its gradients in the method `compute_sos_jacobian()` of the discipline.
- Complete its gradient test (`l1_test_gradient_<model_name>.py`) by also checking for this input/output.

#### I Want to Modify How an Output Is Computed
If you want to modify the way a certain value is computed, it is better to let the possibility to use the old formula instead of losing it.
For that, you can add a flag (a boolean) or a list of options to select how to compute the desired value in the model.
Also, after adding this flag/option, make sure the default parameters are set to compute the value as it was before contributing.
It is only later, during discussions with model reviewers, that the default settings might be changed.

If the output you modify is a coupling variable, also compute its gradients in accordance with the selected option.

**Tests:**
- Find the existing `l0` test of the discipline.
- Create a new test function in the test class and set values to use your new method.
- If the modified output is a coupling variable in Witness:
  - Find the existing `l1` test of the discipline.
  - Create a new test function in the test class and set values to use your new method in the model.
  - Make sure the gradients are OK with the new selected method.

### New Process

You created a new process? Create a `usecase_<my_comment>.py` file that goes in the same folder as the `process.py` file.
Your use case will rely on the process. You can test the use case by creating an instance of your `Study` class and call `my_study.test()` method.

### Existing Process Modifications
This needs further discussions with a Witness model reviewer.
