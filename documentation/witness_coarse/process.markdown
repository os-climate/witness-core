# Table of Contents
1. [WITNESS Coarse Optimisation problem documentation](#witness-coarse-optimisation-problem-documentation)
   1. [Definition](#definition)
   2. [Problem Formulation](#problem-formulation)
      1. [Design Space](#design-space)
      2. [Lower and Upper Bounds](#lower-and-upper-bounds)
      3. [Objectives](#objectives)
         1. [Energy Mean Price](#energy-mean-price)
         2. [Consumption Objective](#consumption-objective)
      4. [Constraints](#constraints)
   3. [Main MDA/MDO Algorithm Parameters](#main-mdamdo-algorithm-parameters)

# WITNESS Coarse Optimisation problem documentation

This process is a version of the Witness coarse problem made with validated models. Witness coarse refers to a reduced complexity of energy models in witness, as opposed to witness full that considers an extended number of energy models.
In witness coarse, energy models are reduced to three categories: fossil, renewable and CCUS (carbon capture and storage).   
The optimization process adjusts the values of the design variables in order to minimize the value of the objective function under given constraints.

## Definition[^1]

Multi-disciplinary design optimization (MDO) is a field of engineering that uses optimization methods to solve design problems incorporating a number of disciplines. It is also known as multidisciplinary system design optimization (MSDO). MDO allows designers to incorporate all relevant disciplines simultaneously in an optimization process. The optimum of the simultaneous problem is more efficient than the design found by optimizing each discipline sequentially, since it can exploit the interactions between the disciplines. However, including all disciplines simultaneously significantly increases the complexity of the problem.


## Problem formulation
The problem is given with an objective to be minimized without constraints. As a result, the optimized solution will reach the lowest objective.

Witness coarse formulation on 22-May-2024 reads:
$$\text{minimize} \quad \text{obj} = \alpha \times \text{energy_{price}_{mean}_{objective}} - (1-\alpha) \times \text{consumption_{objective}}$$
$$\text{wrt} \quad \text{invest_{mix}} \in [1,3000]$$
$$\text{wrt} \quad \text{utilization_{ratios}} \in [1,100]$$

$\alpha$ allows to define the weight of each component of the objective.
Since the $consumption_{objective}$ needs to be maximized and since the global objective function is minimized, $-\text{consumption_{objective}}$ is considered. 
$\alpha$ is chosen so that the weighted components of the objective functions are of the same order of magnitude. 


### Design space
In this optimization process, there are two categories of design variables: the investments and the utilization ratios.
They are all inputs of the five witness coarse energy models, namely:
- **Fossil simple techno**: a simplified model that mimics all fossil energies (gas, fuel, coal, etc.)
- **Renewable simple techno**: a simplified model that mimics all the renewable energies (wind energy, solar energy, etc.) or energy with low carbon footprint such as nuclear energy
- **Carbon capture direct air capture**: CCUS technology that models carbon dioxide capture directly from the air 
- **Carbon capture flue gas capture**: CCUS technology that models carbon dioxide capture directly from flue gas
- **Carbon storage**: CCUS technology that models storage of carbon dioxide that has been captured through the two previous models  

An investment describes the capital invested in one of the five aforementioned technologies during the study period (typically between 2020 and 2100). An investment is > 0. To reduce the complexity of the optimization, the value of the investment is adjusted by the optimizer on a reduced number of years referred to as poles. For instance, if 7 poles are used, investments are optimized for years 2020, 2033, 2046, 2060, 2073, 2086, 2010 and investment values for the remaining years are deduced by b-spline interpolation of the values obtained at the poles. 
An utilization ratio describes the percentage of one of the five aforementioned technologies that is used during the study periof. Its value ranges between 0 and 100. For instance, an utilization ratio of 50% means that the technology is used at 50% of its maximum capacity. Similar to the investments, the utilization ratios are adjusted by the optimizer on a reduced number of poles.
Introducing poles reduces the dimensionality of the optimization problem. For instance if 7 and 11 poles are used for the investments and utilization ratios respectively, then the number of design variables reaches:
$$5 \times \text{models} \times (7 \times \text{poles}_{investments} + 11 \times \text{poles_{utilization}_{ratio}}) = 90 \times \text{design} \quad \text{variables}$$

#### Lower and upper bounds
In the first iterations of the optimization, the L-BGFGS-B optimization algorithm used in the study hits the upper and lower bounds of the design space.
It has been observed that when the lower bounds of the design variables (investments or utilization ratio) are set close to 0 (for instance 1.e-6), then the optimization algorithm has a hard time converging (or does not converge at all).
NB: a lower bound is stricltly positive to avoid computing null gradients which could prevent the optimization from converging.
Chosing lower and upper bounds that are physically realistic has shown to help convergence. For instance, for a net zero emission scenario, typical upper and lower bounds are:
$$10 \leq investment_fossil \leq 3000$$
$$300 \leq investment_renewable \leq 3000$$
$$1 \leq investment_CCUS \leq 3000$$
$$30 \leq utilization_ratio \leq 100$$

It is of utmost importance to check that for the optimized solution, the design variables do not meet the bounds. 
If this case, the bound plays the role of an unwanted constraint that needs to be relieved by considering larger bound values.

Investments are in G<span>$</span> and utilization ratio in percentage.

### Objectives

#### Energy mean price
$$\text{energy_{price}_{mean}_{objective}} = frac{\sum_{years}\text{energy_{price}_{mean}}}{\text{energy_{price}_{ref}} \times n_{years}}$$
where the $\text{energy_{price}_{mean}}[years]$ is the average of the prices of all the energy mix technologies at a given year, namely:
$$\text{energy_{price}_{mean}}[years] = frac{\sum_{technos}\text{energy_{price}}[years]}{n_{technos}}$$
The $\text{energy_{price}_{ref}}$ default value is 100 <span>$</span> so that $\text{energy_{price}_{mean}_{objective}}$ values are around 1.

The energy mean price is affected by the value of the CO2 tax. If the CO2 tax is deactivated, fossil energies are preferred since they are cheaper. 
However, if CO2 tax is activated, renewable energies are preferred as they emit less CO2 and eventually lead to a energy mean price (including CO2 tax) that is lower.

#### Consumption objective
$$\text{consumption_{objective}} = frac{\sum_{years}\text{consumption}[years]}{\text{consumption_{ref}} \times n_{years}}$$
As defined in the documentation of the macroeconomics discipline, consumption C is the part of the net output not invested, namely:
$$C = Q - I$$
where Net output $Q$ is the output net of climate damage explained in the macroeconomics discipline documentation.

The $\text{consumption_{ref}}$ default value is 250 T\$ so that $\text{consumption_{objective}}$ values are around 1.
In witness, it is assumed that the larger the consumption per capita, the better the wealth of the population over the entire period of the study, which is what is aimed at. The consumption objective is therefore a quantity to be maximized.

From the equation above, one could think that reducing the investments (I) would maximize the consumption (C). However, reducing the investments in energy also reduces the net output (see the impact of energy investments on usable capital Ku in the macroeconomics documentation).
Because of this coupling between energy and economy, finding the optimum investments is not straightforward.

### Constraints
No equality or inequality constraint has been activated in this optimization.

## Main MDA/MDO algorithm parameters

|      **Parameter**      |     **Value**     |
|:-----------------------:|:-----------------:|
|      MDA algorithm      | MDA Gauss-Seidel  |
| Differentiation method  | Analytical (user) |
| Linear solvers MDA/MDO  |    GMRES-PETSC    |
| MDA/MDO preconditioners |       gasm        | 
|      MDO algorithm      |     L-BFGFS-B     |
|  Linearization method   |      Adjoint      | 


[^1]: Wikipedia - Multidisciplinary design optimization - Retrieved from: 'https://en.wikipedia.org/wiki/Multidisciplinary_design_optimization'