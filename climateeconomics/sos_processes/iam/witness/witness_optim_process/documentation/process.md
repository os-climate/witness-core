# WITNESS Optimisation problem documentation

This process is a converged version of the full Witness problem made with validated models. It is used to refine optimization problem coefficients and demonstration.

## Definition[^1]

Multi-disciplinary design optimization (MDO) is a field of engineering that uses optimization methods to solve design problems incorporating a number of disciplines. It is also known as multidisciplinary system design optimization (MSDO). MDO allows designers to incorporate all relevant disciplines simultaneously in an optimization process. The optimum of the simultaneous problem is more efficient than the design found by optimizing each discipline sequentially, since it can exploit the interactions between the disciplines. However, including all disciplines simultaneously significantly increases the complexity of the problem.


## Problem formulation

The problem is given with an objective to minimize, respecting a certain number of constraints. As a result, the optimized solution will attend the lowest objective, following the given constraints.

The problem is defined as follow : 

![](witness_formulation.PNG)



## Objectives

The objective of the optimisation is:
$$Objective = utility\_objective + min\_utility\_objective + carbon\_objective$$
 The weight of each component of the objective is defined by $\alpha$ and $\gamma$ .
$\alpha$ represents if the effort is put on utility (global wealth) or low CO2 emission.
$\gamma$ represents if we prefer maximum average utility or do not fall to low at the minimum.


### Utility

Utility represents the global wealth of the population. The physical meaning of the utility objective is to maximise the average wealth of the population over the entire period of the study. 

Last utility objective, formula:

$$utility\_objective =\alpha\frac{initial\_utility}{last\_utility}$$  

Welfare objective, formula:

$$utility\_objective =\alpha\frac{initial\_discounted\_utility * Δyears}{welfare}$$  

### Min Utility

As utility is the wealth of the population, it is important it remains likely stable, or at least there is no catastrophic years where utility is near zero. In order to assure this,an objective is given on the minimum utility : the global wealth should not dramatically fall.

Last min_utility objective, formula:

$$min\_utility\_objective =(1 - α)\frac{initial\_min\_utility}{last\_min\_utility}$$  



### Carbon Emission

Carbon emissions are central for sustainability. This objective tends to reduce the CO2 emissions by wisely choosing energy production technologies.

The formula of carbon emission objective is the following: 
$$carbon\_objective = 0.5*(1 - α)\frac{ \sum{carbon\_emissions} }{initial\_carbon\_emissions*Δyears}$$

## Boundaries

As the problem is numerical and resolved by computer, the research domain can not be infinite. As a result, bounds are set to limit it, but are kept large enough to do not affect the result.
* invest_mix is set between 0% and 100%. 

## Constraints

As the problem is resolved numerically, a great number of solutions can be found, including non realistic ones. In order to keep solution as realistic as possible, constraints are set.
Here is the physical meaning of the constraints : 

### Total energy production
The total energy production is one of the outputs of the witness process. This represents the total amount of energy produced. As our society strongly rely on energy production (electricity, transport, gas,...) we do not want energy production to fall under a certain amount min_energy.

### Energy net production
For each energy considered (methane, electricity, hydrogen ...), the amount of energy produced has to be higher than the demand of this energy.

### Liquid_fuel, H2 and H2 liquid
These 3 types of energy are the main resources used in the transport field. As our society strongly rely on transport,  a minimum production is set to supply the transport field. As example, turning all energy production into nuclear production is unrealistic, because there would be no solution for cars, planes, ships... which is non-realistic.

### Carbon consumption
Different technologies use CO2 as a resource. But this CO2 needs to be stored in order to be used. In other words this constrains assures that CO2 used is possessed.

### Hydropower production
It is assumed that all water flows for hydropower are exploited in 2020. Thus, there is no way to increase the energy production by hydropower.

### Solid fuel, electricity and biomass
With the same logic than the constraint about energy for transport, the society needs a certain amount of energy for houses and buildings.

### H2 liquid production
Gaseous hydrogen is difficult to transport and stock, while it is easier with liquid hydrogen. That is why liquid hydrogen is privileged.

### Land use
Witness model computes land use by technologies and agriculture. This constraints assures 
that the land used is available, and does not exceed forest and crops land limits.

## Design variables
Design variables are variables that the optimizer is able to change in order to find the best solution to the given problem. they are degrees of freedom of the problem.
The following parts describes the different design variables of the problem.

### Investment in the energy mix
The optimizer is able to change the investment in the energy mix, thus, the investment in each energy and each technology. This allows to explore all the energy production scenario, regarding the constraints describes previously.

### Investment in CCS
Following the same idea than the investment in the energy mix, the optimizer is able to change the investment in carbon capture and storage technologies.

### Percentage of CCS
This design variable permits to split the global investment into energy production and CO2 capture and storage.

### Livestock usage
This design variable represent the percentage of utilisation of the agriculture surface dedicated to livestock. This permits to mimic a reduction of meet production and so to reduce the land surface used by agriculture.


[^1]: Wikipedia - Multidisciplinary design optimization - Retrieved from: 'https://en.wikipedia.org/wiki/Multidisciplinary_design_optimization'