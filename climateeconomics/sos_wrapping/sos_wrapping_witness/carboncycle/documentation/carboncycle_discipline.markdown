## Carbon cycle

For the carbon cycle model we use the one from DICE (Nordhaus, 2017[^1]) without any modification.
It uses a three-reservoir model calibrated with historical data. The three reservoirs are the deep ocean, the upper ocean and the atmosphere. Carbon flows in both directions between the reservoirs. $M_{t\,i}$ represents the mass of carbon in gigatons in the reservoir $i$ at period $t$:
$$M_{t+1\,AT} = M_{t\,AT}.b11 + M_{t\,UP}.b21 + E_t.\frac{time\,step}{3.666}$$
$$M_{t+1\,UP}= M_{t\,AT}.b12 + M_{t\,UP}.b22 + M_{t\,LO}.b32$$
$$M_{t+1\,LO}= M_{t\,LO}.b33 + M_{t\,UP}.b23$$

with $AT$ the atmosphere reservoir, $UP$ the upper ocean and $LO$ the deeper ocean. The parameters $b_{ij}$ represents the flows between the reservoirs.

The model only considers CO2, other greenhouse gases are assumed to be exogenous and enter the forcing equation explained in $Temperature$ model.

### Model inputs and outputs

The standard inputs are:

- year start, the starting year of the study. Default is 2020.
- year end, the last year of the study. Default is 2100.
- time step, the number of year between each step of computation. Default is 1.
- CO2 emissions, the quantity of CO2 released in the atmosphere in Gt. "total_emissions" column gives the emissions each year. "cum_total_emissions" is the cumulative emission of land and industry. This data comes from Carbon_emission model.
- Alpha, is the trade variable between utility and CO2 emission, used to compute the output ppm objective. The weight of utility is Alpha, the weight of climate is (1 - Alpha). The default value is 0.5.
- Beta is the trade variable between CO2 emission and temperature, used to compute the output ppm objective. The weight of CO2 emission is Beta, the weight of temperature is (1 - Beta).

Advanced inputs are accessible to advanced and expert user only. Advanced inputs are:

- Equilibrium atmospheric concentration, the atmospheric concentration of carbon at the equilibrium. Unit is Giga-tons of carbon (Gtc). Default value is 558 Gtc.
- Lower strata equilibrium concentration, the concentration of carbon at the equilibrium in the lower strata of the atmosphere. Unit is Gtc. Default value is 1720 Gtc.
- Upper strata equilibrium concentration, the concentration of carbon at the equilibrium in the upper strata of the atmosphere. Unit is Gtc. Default value is 360 Gtc.
- Initial atmospheric concentration, the initial atmospheric concentration of carbon in GtC. Unit is Gtc. Default value is 878.412 Gtc.
- Initial atmospheric concentration in lower strata, the initial atmospheric concentration of carbon in the lower strata of the atmosphere. Unit is Gtc. Default value is 1740 Gtc.
- Initial atmospheric concentration in upper strata, the initial atmospheric concentration of carbon in the upper strata of the atmosphere. Unit is Gtc. Default value is 460 Gtc.
- Atmospheric concentration lower bound, the lower limit of atmospheric concentration of carbon. Unit is Gtc. Default value is 10 Gtc.
- Deep ocean concentration lower bound, the lower limit of carbon concentration in deep ocean. Unit is Gtc. Default is 1000 Gtc.
- Shallow ocean concentration lower bound, the lower limit of carbon concentration in shallow ocean. Unit is Gtc. Default is 100 Gtc.

Expert inputs are accessible to expert user only. Expert inputs are:

- Carbon cycle transition matrix parameter b_12, is a parameter to compute the transition of carbon, mainly around upper ocean strata. Default value is 0.12.
- Carbon cycle transition matrix parameter b_23, is a parameter to compute the transition of carbon, mainly around lower ocean strata. Default value is 0.007.

The outputs of the models are:

- ppm_objective, gives the concentration of carbon of the atmosphere in parts per millions for each year.
- Carbon cycle data, gives the concentration of carbon in the atmosphere for each year.
- Carbon_cycle_detail_df, gives the concentration of carbon in atmoshpere and ocean.

### PPM objective

A "ppm objective" is calculated and set as an output of the model, based on the number of CO2 particule per meter:
$$ppm_{objective} = \frac{(1 - \alpha) * (1 - \beta) * \sum CO2_{ppm} }{(CO2_{ppm}^{ref} * \Delta_{years})}$$

where $CO2_{ppm}^{ref}$ is a reference value used to normalize the value of the objective, $\beta$ is a trade variable between the objectives based on the CO2 emissions or concentration and $\alpha$ is the global tradeof variable between global warning and the economy.

## References

[^1]: Nordhaus, W. D. (2017). Revisiting the social cost of carbon. Proceedings of the National Academy of Sciences, 114(7), 1518-1523.
