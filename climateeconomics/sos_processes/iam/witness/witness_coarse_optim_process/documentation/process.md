# WITNESS Coarse Optimisation problem documentation

This process is a version of the Witness coarse problem made with validated models. Witness coarse refers to a reduced complexity of energy models in witness, as opposed to witness full that considers an extended number of energy models.
In witness coarse, energy models are reduced to three categories: fossil, renewable and CCUS (carbon capture and storage).   
The optimization process adjusts the values of the design variables in order to minimize the value of the objective function under given constraints.

## Definition[^1]

Multi-disciplinary design optimization (MDO) is a field of engineering that uses optimization methods to solve design problems incorporating a number of disciplines. It is also known as multidisciplinary system design optimization (MSDO). MDO allows designers to incorporate all relevant disciplines simultaneously in an optimization process. The optimum of the simultaneous problem is more efficient than the design found by optimizing each discipline sequentially, since it can exploit the interactions between the disciplines. However, including all disciplines simultaneously significantly increases the complexity of the problem.


## Problem formulation

The problem is given with an objective to be minimized without constraints. As a result, the optimized solution will reach the lowest objective.

The problem is defined as follows: 
$$minimize obj = $\alpha$ \times frac{\sum_{years}energy\_mean\_price[y]}{energy\_price\_ref \times n\_years} + \gamma \times frac{\sum_{years}consumption[y]}{consumption\_ref \times n\_years}$$
wrt $invest\_mix \in [1,3000]$
wrt $utilization\_ratios \in [30,100]$

## Design space
In this optimization process, there are two categories of design variables: the investments and the utilization ratios.
They are all inputs of the five witness coarse energy models, namely:
- **Fossil simple techno**: a simplified model that gathers all fossil energies (gas, fuel, coal, etc.)
- **Renewable simple techno**: a simplified model that gathers all the renewable energies (wind energy, solar energy, etc.) or energy with low carbon footprint such as nuclear energy
- **Carbon capture direct air capture**: CCUS technology that models carbon dioxide capture directly from the air 
- **Carbon capture flue gas capture**: CCUS technology that models carbon dioxide capture directly from flue gas
- **Carbon storage**: CCUS technology that models storage of carbon dioxide that has been captured through the two previous models  

An investment describes the capital invested in one of the five aforementioned technologies between 2020 and 2100. An investment is > 0. To reduce the complexity of the optimization, the value of the investment is adjusted by the optimizer for 7 years, the so-called poles (2020, 2033, 2046, 2060, 2073, 2086, 2010). Then, the value of the investments for the 73 remaining years are obtained by b-spline interpolation of the values obtained for the poles.   
An utilization ratio describes the percentage of one of the five aforementioned technologies that is used between 2020 and 2100. Its value ranges between 0 and 100. For instance, an utilization ratio of 50% means that the technology is used at 50% of its maximum capacity. Similar to the investments, the utilization ratios are adjusted by the optimizer for 11 years (2020, 2028, 2037, 2046, 2055, 2064, 2073, 2081, 2091, 2100). A b-spline interpolation of the values obtained at the poles provides the values for the missing years.

Eventually, the number of design variables reaches:
$$5 \times models \times (7 \times poles\_investments + 11 \times poles\_utilization\_ratio) = 90 \times design variables$$

### Lower and upper bounds
In the first iterations of the optimization, the L-BGFGS-B optimization algorithm used in the study hits the upper and lower bounds of the design space.
It has been observed that when the lower bounds of the design variables (investments or utilization ratio) are set to 0 or very close to it (for instance 1.e-6), then the optimization algorithm has a hard time converging (or does not converge at all).
Chosing lower and upper bounds that are physically realistic proves to help convergence. For instance, for a net zero emission scenario, typical upper and lower bounds are:
$$10 \leq investment\_fossil \leq 3000$$
$$300 \leq investment\_renewable \leq 3000$$
$$1 \leq investment\_CCUS \leq 3000$$
$$30 \leq utilization\_ratio \leq 100$$

Investments are in G<span>$</span> and utilization ratio in percentage.

## Objectives

The optimisation objective can be rewritten as:
$$Objective = \alpha \times energy\_mean\_price\_objective + \gamma \times consumption\_objective$$

$\alpha$ and $\gamma$ define the weight of each component of the objective.
Since the $consumption\_objective$ needs to be maximized and since the global objective function is minimized, $\gamma < 0$. 
Then, the ratio between $\alpha$ and $\gamma$ is chosen so that the weighted components of the objective functions are of the same order of magnitude. 

### Energy mean price
$$energy\_mean\_price\_objective = frac{\sum_{years}energy\_mean\_price}{energy\_price\_ref \times n\_years}$$
where the $energy\_mean\_price[y]$ is the average of the prices of all the energy mix technologies at a given year, namely:
$$energy\_mean\_price[y] = frac{\sum_{technos}energy\_price[y]}{n\_technos}$$
The $energy\_price\_ref$ default value is 100 <span>$</span> so that $energy\_mean\_price\_objective$ values are around 1.


### Consumption objective
$$consumption\_objective = frac{\sum_{years}consumption}{consumption\_ref \times n\_years}$$
As defined in the documentation of the macroeconomics discipline, consumption C is the part of the net output not invested, namely:
$$C = Q - I$$
where Net output $Q$ is the output net of climate damage explained in the macroeconomics discipline documentation.

The $consumption\_ref$ default value is 250 T\$ so that $consumption\_objective$ values are around 1.
In witness, it is assumed that the larger the consumption per capita, the better the wealth of the population over the entire period of the study, which is what is aimed at. The consumption objective is therefore a quantity to be maximized.

## Constraints
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