# Demand Discipline

## Role

Generation of a demand in $ for each sector based on :
- population
- a value of reference for yearly need in each sector, in $/person. For example, GDP Agriculture 2020 / GDP 2020. 

For each sector, demand, expressed in $, is calculated as : 
$$ demand\ sector\ S = population\ x Value\ of \reference \sector S $$

## Redistribution of resources

#### Energy

Inputs :
- Total energy production (out of EnergyMixDiscipline)
- Share (%) of the total energy production attributed to each sector

Output:
- for each sector :

$$ allocated\ energy\ for\ sector\ S = Share\ energy\ sector\ S x Total\ energy\ production S$$


#### Investments

Inputs :
- Investments [Trillion $] for sector S (for each sector, fixed or design variable)

Output:
- Total investments 

$$ Total\ investments\ = Sum\ of\ investments\ on\ sectors\ $$

