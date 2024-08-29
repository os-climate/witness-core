# Consumption Discipline

## Role

Generation of the consumption in $ for each sector based on :
- population
- a value of reference for yearly need in each sector, in $/person. For example, GDP Agriculture 2020 / GDP 2020. 

For each sector, consumption, expressed in $, is calculated as :
$$consumption\ sector\ S = population \times Value\ of \reference \sector S$$

## Redistribution of resources

#### Energy

Inputs :
- Total energy production (out of EnergyMixDiscipline)
- Share (%) of the total energy production attributed to each sector

Output:
- for each sector :

$$allocated\ energy\ for\ sector\ S = Share\ energy\ sector\ S \times Total\ energy\ production\ S$$

- Total :

$$Total\ allocated\ energy\ = Sum\ of\ allocated\ energy\ on\ sectors$$

#### Investments

Inputs :
- Investments [Trillion $] for sector S (for each sector, fixed or design variable)

Output:
- Total investments 

$$Total\ investments\ = Sum\ of\ investments\ on\ sectors\$$

