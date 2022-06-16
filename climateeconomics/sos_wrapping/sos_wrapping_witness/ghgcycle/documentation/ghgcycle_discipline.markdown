## Greenhouse Gas cycle 

For the Greenhouse Gas cycle model we use the one from FUND. 

![Fund.PNG](fund.PNG)

The model is due to Maier-Reimer, E. and K.Hasselmann (1987).[^1]
The model parameters are from Hammitt, J.K., R.J.Lempert, and M.E.Schlesinger (1992).[^2]
You can find an implementation (and parameters) of this model on GitHub.[^3]

### Model inputs and outputs

The inputs are:
* year start, the starting year of the study. Default is 2020.
* year end, the last year of the study. Default is 2100.
* time step, the number of year between each step of computation. Default is 1.
* GHG emissions, the quantity of GHG released in the atmosphere in Gt. This data comes from ghg_emission model. 
* Alpha, is the trade variable between utility and CO2 emission, used to compute the output ppm objective. The weight of utility is Alpha, the weight of climate is (1 - Alpha). The default value is 0.5.
* Beta is the trade variable between CO2 emission and temperature, used to compute the output ppm objective. The weight of CO2 emission is Beta, the weight of temperature is (1 - Beta).
* Numerical parameters from the FUND model. In paricular, emissions fractions, decay rates and initial concentrations for CO2, CH4 and N2O gases.

The outputs of the models are:
* ppm\_objective, gives the concentration of carbon of the atmosphere in parts per millions for each year.
* GHG cycle data, gives the concentration of carbon in the atmosphere for each year.
* GHG\_cycle\_detail\_df, gives the concentration of carbon in atmoshpere and ocean. 

### PPM objective

A "ppm objective" is calculated and set as an output of the model, based on the number of CO2 particule per meter:
$$ppm_{objective} = \frac{(1 - \alpha) * (1 - \beta) * \sum CO2_{ppm} }{(CO2_{ppm}^{ref} * \Delta_{years})}$$

where $CO2_{ppm}^{ref}$ is a reference value used to normalize the value of the objective, $\beta$ is a trade variable between the objectives based on the CO2 emissions or concentration and $\alpha$ is the global tradeof variable between global warning and the economy.
## References 
[^1]: Maier-Reimer, E. and K.Hasselmann (1987), 'Transport and Storage of Carbon Dioxide in the Ocean: An Inorganic Ocean Circulation Carbon Cycle Model', Climate Dynamics, 2, 63-90.
[^2]: Hammitt, J.K., R.J.Lempert, and M.E.Schlesinger (1992), 'A Sequential-Decision Strategy for Abating Climate Change', Nature, 357, 315-318.
[^3]: FUND Repository on GitHub (https://github.com/fund-model/MimiFUND.jl/tree/master/src)