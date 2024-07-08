# Table of Contents
1. [WITNESS Coarse Optimisation problem documentation](#witness-coarse-optimisation-problem-documentation)
   1. [Definition](#definition)
   2. [Problem Formulation](#problem-formulation)
      1. [Design Space](#design-space)
      2. [Lower and Upper Bounds](#lower-and-upper-bounds)
      3. [Objective Population Utility](#objective-population-utility)
         1. [Utility per capita](#utility-per-capita)
            1. [Consumption per capita in Witness](#consumption-per-capita-in-witness)
            2. [Energy price in Witness](#energy-price-in-witness)
      4. [Anti decreasing net GDP objective](#anti-decreasing-net-gdp-objective)
      5. [Constraints](#constraints)
   4. [Main MDA/MDO Algorithm Parameters](#main-mdamdo-algorithm-parameters)

# WITNESS Coarse Optimisation problem documentation

This process is a version of the Witness coarse problem made with validated models. Witness coarse refers to a reduced complexity of energy models in witness, as opposed to witness full that considers an extended number of energy models.
In witness coarse, energy models are reduced to three categories: fossil, renewable and CCUS (carbon capture and storage).   
The optimization process adjusts the values of the design variables in order to minimize the value of the objective function under given constraints.

## Definition[^1]

Multi-disciplinary design optimization (MDO) is a field of engineering that uses optimization methods to solve design problems incorporating a number of disciplines. It is also known as multidisciplinary system design optimization (MSDO). MDO allows designers to incorporate all relevant disciplines simultaneously in an optimization process. The optimum of the simultaneous problem is more efficient than the design found by optimizing each discipline sequentially, since it can exploit the interactions between the disciplines. However, including all disciplines simultaneously significantly increases the complexity of the problem.


## Problem formulation
The problem is given with an objective to be minimized without constraints. As a result, the optimized solution will reach the lowest objective.

Witness coarse formulation on 22-May-2024 reads:
$$minimize \quad obj = \alpha_1 \text{Population utility} + \alpha_2 \text{Anti-decreasing net GDP} + \alpha_3 \text{Energy Wasted Objective}$$
$$\text{wrt }\text{design variables} \in \text{design space}$$

The $\alpha$'s are the weights before each objective. Their values are:-
1. $\alpha_1=1$, objective should be minimized
2. $\alpha_2=-1$, objective should be maximized
3. $\alpha_3=0.1$, objective is of a reduced importance and  should be minimized.

More details on the definitions of the objectives can be found in later sections.

### Design variables
In this optimization process, there are two categories of design variables: the investments and the utilization ratios.
They are all inputs of the five witness coarse energy models, namely:

#### Design variables controling energy and CCUS techno production

##### Energy technos:
The energies in witness coarse are 
- **Fossil simple techno**: a simplified model that mimics all fossil energies (gas, fuel, coal, etc.)
- **Renewable simple techno**: a simplified model that mimics all the renewable energies (wind energy, solar energy, etc.) or energy with low carbon footprint such as nuclear energy

##### CCUS technos
- **Carbon capture direct air capture**: CCUS technology that models carbon dioxide capture directly from the air 
- **Carbon capture flue gas capture**: CCUS technology that models carbon dioxide capture directly from flue gas
- **Carbon storage**: CCUS technology that models storage of carbon dioxide that has been captured through the two previous models  

For each techno there are design variables controling 
- The investement : a investement leads to a given maximal production capacity (mimics building new plants of the techno).
- An utilization ratio : describes the intensity, in percent, at which the techno is used. For instance, an utilization ratio of 50% means that the technology is used at 50% of its maximum capacity.
Introducing poles reduces the dimensionality of the optimization problem. For instance if 7 and 11 poles are used for the investments and utilization ratios respectively, then the number of design variables reaches:
$$5 \times models \times (7 \times poles_{investments} + 11 \times poles_{utilization \textunderscore ratio}) = 90 \times design \textunderscore variables$$

#### Lower and upper bounds
In the first iterations of the optimization, the L-BGFGS-B optimization algorithm used in the study hits the upper and lower bounds of the design space.
It has been observed that when the lower bounds of the design variables (investments or utilization ratio) are set close to 0 (for instance 1.e-6), then the optimization algorithm has a hard time converging (or does not converge at all).
NB: a lower bound is stricltly positive to avoid computing null gradients which could prevent the optimization from converging.
Chosing lower and upper bounds that are physically realistic has shown to help convergence. For instance, for a net zero emission scenario, typical upper and lower bounds are:
$$10 \leq investment \textunderscore fossil \leq 3000$$
$$300 \leq investment \textunderscore renewable \leq 3000$$
$$1 \leq investment \textunderscore CCUS \leq 3000$$
$$30 \leq utilization \textunderscore ratio \leq 100$$

It is of utmost importance to check that for the optimized solution, the design variables do not meet the bounds. 
If this case, the bound plays the role of an unwanted constraint that needs to be relieved by considering larger bound values.

Investments are in G<span>$</span> and utilization ratio in percentage.

Better managing the bounds could also potentially improve the mda convergence. 
Indeed, in the GS pure Newton mda algorithm, a gradient is computed using the same analytical formulas as for the MDO. 
Matrix inversion can be difficult and if preconditionning does not help, the gradients at the bounds could be a root cause of the issue.

### Objective Population Utility

$$\text{maximize}_{x \in \text{design space}} \text{ Population utility objective (x)}$$


The `Population utility objective` (a float) is the average over the years of utility of the population.
$$\text{Population utility objective} = \frac{1}{\text{nb years}}\sum_{\text{year in years}} \text{Population utility (year)}$$

with 

$$\text{Population utility} = \frac{\text{Population}}{\text{Population at year start}} \text{Utility per capita}$$

The next section described the notion of  *Utility per capita*.

#### Utility per capita

The utility per capita relies on two variables available in witness, *Consumption per capita* and *Energy price*. The next two sections gives a quick explanation of these variables.

In our optimization formulation, we want to maximize the quantity of things consumed. For that, we can see *Consumption per capita* can be seen as 

$$C^{pc} = Q^{pc} \times P$$

that is, a quantity (of "things" consumed) $\times$ Price ("average price of things consumed"). 
The assumption we make is that the average price of things that are consumed is driven by energy price, leading to :


$$\text{quantity per capita} = \frac{\text{consumption per capita}}{\text{energy price}}$$

If we take year start as a reference point, and apply a function $f$ to mimic saturation to consumption (having more when your poor is huge, but having more when you already have a lot doesnt mean much to you), we defined the gain of utility as

$$\text{utility per capita (year)} = f \left(\frac{\text{quantity per capita (year)}}{\text{quantity per capita (year start)}} \right)$$

> This saturation function is an S-curve, whose parameters have been fine-tuned, but can be tweeked based on your preferences. 

##### Consumption per capita in Witness

Consumption $C$ is the part of the net output not invested, namely:
$$C = Q - I$$
where Net output $Q$ is the output net of climate damage explained in the macroeconomics discipline documentation.

From the equation above, one could think that reducing the investments (I) would maximize the consumption (C). However, reducing the investments in energy also reduces the net output (see the impact of energy investments on usable capital Ku in the macroeconomics documentation).
Because of this coupling between energy and economy, finding the optimum investments is not straightforward, therefore showing the need for an optimizer to find the perfect balance.

Consumption per capita $C^{pc}$ is simply the $C$ divided by the population.

##### Energy price in witness

The energy price in Witness is the average of the prices of all the energy mix technologies at a given year, namely:
$$energy \textunderscore price_{mean}[years] = \frac{1}{n_{technos}} \times \sum_{technos}energy \textunderscore price[years, technos]$$

The energy price is affected by the value of the CO2 tax. If the CO2 tax is deactivated, fossil energies are preferred since they are cheaper. 
However, if CO2 tax is activated, renewable energies are preferred as they emit less CO2 and eventually lead to a energy mean price (including CO2 tax) that is lower.


### Anti decreasing net GDP objective

To prevent an edge effect at end of simulation, we introduce this anti-decreasing objective for the GDP net of damages, denoted by $Q$.
Indeed, the population utility objective mentioned above tends to maximize the integral of utility per capita, causing most of the time a peak at mid-scenario, and a sudden drop at the very end, to maximize surface. The objective introduce here is there to smooth the curve, and prevent this edge effect.

$$\text{anti decreasing net GDP obj} = \frac{\sum_i \left( \min\left(\frac{Q_{i+1}}{Q_i}, 1\right) - 1 \right)}{\text{nb years}}$$

where $Q_i$ represents the GDP net of damage at year $i$. This function captures the relative change in GDP over successive years, focusing on periods of decline. By utilizing the minimum function, $\min\left(\frac{Q_{i+1}}{Q_i}, 1\right)$, the objective ensures that only negative or no growth affects the result, normalizing any growth periods to zero.

An important feature of this objective function is its self-normalization within the range $[0, 1]$, obviating the need for any external reference values. The value of the function should be minimized, reflecting the goal of reducing the frequency and magnitude of GDP decline.

### Constraints
No equality or inequality constraint has been activated in this optimization.

## Main MDA/MDO algorithm parameters

Robustness of MDA/MDO can be impacted by the choice of some parameters. For instance, solving the MDA with a Gauss-Seidel 
approach can require several times less memory than a Gauss Seidel pure Newton approach. This could lead to out of memory issues. 
This is due to the need to invert a matrix in the Newton approach as already discussed earlier in the section "lower and upper bounds"
Precondionning can help as well as the linear solver used, this is why those parameters are noted in the table.
Then, some optimization algorithms can hit the bounds at the beginning of the optimization which can lead to convergence issues as discussed earlier.
The algorithm chose is therefore reminded in the table below.

|      **Parameter**      |     **Value**     |
|:-----------------------:|:-----------------:|
|      MDA algorithm      | MDA Gauss-Seidel  |
| Differentiation method  | Analytical (user) |
| Linear solvers MDA/MDO  |    GMRES-PETSC    |
| MDA/MDO preconditioners |       gasm        | 
|      MDO algorithm      |     L-BFGFS-B     |
|  Linearization method   |      Adjoint      | 


[^1]: Wikipedia - Multidisciplinary design optimization - Retrieved from: 'https://en.wikipedia.org/wiki/Multidisciplinary_design_optimization'
